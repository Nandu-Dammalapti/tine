import asyncio
import json
import logging

import httpx
import pytest

from tine.ledger import Ledger
from tine.policy import DIGEST_FIELD, FREEZE_EDITED, FREEZE_UNCHANGED, policy_digest
from tine.proxy import ProxyConfig, create_app
from tine.shadow import ShadowStore, extract_answer, sampled, shadow_stages, validate_shadow_policy

POLICY = {"version": 1, "tenant": "a", "frontier": "frontier", "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
          "stages": [{"models": ["small-a", "small-b"], "check": "agreement"}]}
PRICES = {f"openai:{name}": {"input_per_million": rate, "output_per_million": rate}
          for name, rate in (("frontier", 100.0), ("small-a", 1.0), ("small-b", 2.0))}


def config(**changes):
    values = dict(tenant_keys={"token-a": "a", "token-b": "b"}, openai_api_key="secret",
                  openai_frontier_model="frontier", prices=PRICES, verified_cache=False,
                  shadow_policies={"a": {"openai": POLICY}}, shadow_sample_rate=1.0, shadow_max_cost_usd=1.0)
    values.update(changes)
    return ProxyConfig(**values)


def reply(text):
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}


def serve(answers, seen, fail=(), delay=0.0, unmetered=()):
    async def provider(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if delay and model != "frontier":
            await asyncio.sleep(delay)
        if model in fail:
            return httpx.Response(500, json={"error": "down"})
        if model in unmetered:  # a reply that does not say what it used: its cost cannot be known
            return httpx.Response(200, json={"choices": reply(answers[model])["choices"]})
        return httpx.Response(200, json=reply(answers[model]))
    return provider


def ask(app_config, provider, token="token-a", count=1, together=False, extra=None):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(app_config, ledger=Ledger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                headers = {"authorization": f"Bearer {token}"}
                posts = [client.post("/v1/chat/completions", headers=headers,
                                     json={"model": "frontier", "messages": [{"role": "user", "content": f"q{i}"}],
                                           **(extra or {})})
                         for i in range(count)]
                answers = await asyncio.gather(*posts) if together else [await post for post in posts]
                report = await client.get("/v1/shadow", headers=headers)
                health = await client.get("/health")
                return answers, report, health
    return asyncio.run(scenario())


def test_shadow_runs_after_the_answer_and_reports_what_the_policy_would_have_done():
    seen = []
    answers, report, health = ask(config(), serve({"frontier": "B", "small-a": " B", "small-b": "B "}, seen))
    assert answers[0].status_code == 200 and answers[0].json() == reply("B")  # the caller gets the frontier's answer
    assert seen == ["frontier", "small-a", "small-b"] and health.json()["shadow"] == "sampling"
    body = report.json()
    assert body["shadowed_requests"] == 1 and body["policy_would_accept"] == 1
    assert body["accepted_matching_frontier_share"] == 1.0
    # 2,000 tokens each: frontier $0.20; the two cheap calls $0.002 + $0.004.
    assert body["frontier_cost_usd"] == pytest.approx(0.2)
    assert body["projected_policy_cost_usd"] == pytest.approx(0.006)
    assert body["projected_savings_fraction"] == pytest.approx(0.97)
    assert body["shadow_spend_usd"] == pytest.approx(0.006)


def test_disagreement_projects_the_frontier_cost_on_top_and_failures_stay_contained():
    seen = []
    answers, report, _ = ask(config(), serve({"frontier": "B", "small-a": "A", "small-b": "B"}, seen))
    body = report.json()
    assert body["policy_would_accept"] == 0 and body["accepted_matching_frontier_share"] is None
    assert body["projected_policy_cost_usd"] == pytest.approx(0.206)
    assert body["projected_savings_fraction"] == pytest.approx(1 - 0.206 / 0.2)  # negative: escalation costs more
    seen = []
    answers, report, _ = ask(config(), serve({"frontier": "B", "small-a": "B", "small-b": "B"}, seen, fail={"small-b"}))
    assert answers[0].status_code == 200
    body = report.json()
    assert body["policy_would_accept"] == 0 and body["failed_shadow_calls"] == 1
    assert body["unknown_cost_requests"] == 1 and body["projected_savings_fraction"] is None


def test_shadow_respects_the_tenant_the_sample_rate_and_the_spending_cap():
    seen = []
    ask(config(), serve({"frontier": "B", "small-a": "B", "small-b": "B"}, seen), token="token-b")
    assert seen == ["frontier"]  # tenant b has no policy
    seen = []
    _, report, _ = ask(config(shadow_max_cost_usd=0.01), serve({"frontier": "B", "small-a": "B", "small-b": "B"}, seen),
                       count=4)
    assert report.json()["shadowed_requests"] == 2  # $0.006 each: the third would start above the cap
    assert not sampled("any", 0.0) and sampled("any", 1.0)
    share = sum(sampled(f"request-{i}", 0.25) for i in range(4000)) / 4000
    assert 0.2 < share < 0.3


def test_shadow_configuration_is_checked_before_anything_runs():
    for changes in [{"shadow_max_cost_usd": None}, {"shadow_sample_rate": 2},
                    {"shadow_policies": {"nobody": {"openai": POLICY}}},
                    {"shadow_policies": {"a": {"openai": {**POLICY, "frontier": "other"}}}},
                    {"shadow_policies": {"a": {"openai": {**POLICY, "tenant": "b"}}}},
                    {"shadow_policies": {"a": {"openai": {**POLICY, "stages": [{"models": ["x"], "check": "tests+judge"}]}}}},
                    {"answer_pattern": "("}]:
        with pytest.raises(ValueError):
            create_app(config(**changes), ledger=Ledger())
    with pytest.raises(ValueError, match="No check is configured"):
        validate_shadow_policy({**POLICY, "stages": [{"models": ["x"], "check": "tests+judge"}]}, "frontier")


def test_answers_are_extracted_by_pattern_and_shared_models_are_called_once():
    payload = reply("I considered (A) but the answer is (C).")
    assert extract_answer("openai", payload, r"\(([A-Z])\)") == "C"
    assert extract_answer("openai", payload, r"\[(\d)\]") is None and extract_answer("openai", {"choices": []}) is None
    claude = {"content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "42"}]}
    assert extract_answer("anthropic", claude) == "42"
    policy = {**POLICY, "stages": [{"models": ["a", "b"], "check": "agreement"}, {"models": ["b", "c"], "check": "agreement"}]}
    made = []

    async def call(model):
        made.append(model)
        return {"answer": {"a": "1", "b": "2", "c": "2"}[model], "cost_usd": 1.0}
    outcome = asyncio.run(shadow_stages(policy, call))
    assert sorted(made) == ["a", "b", "c"] and outcome["accepted"] and outcome["stage"] == 1 and outcome["served_by"] == "b"


def test_shadow_store_keeps_tenants_apart_and_stores_no_text():
    ledger = Ledger()
    store = ShadowStore(ledger)
    outcome = {"accepted": True, "stage": 0, "served_by": "small-a", "answer": "secret answer",
               "calls": [{"model": "small-a", "cost_usd": 0.5, "error": None}]}
    store.record(request_id="r1", tenant="a", provider="openai", policy=POLICY, outcome=outcome,
                 frontier_answer="secret answer", frontier_cost_usd=2.0)
    assert store.report("a")["shadowed_requests"] == 1 and store.report("b")["shadowed_requests"] == 0
    assert store.spend("a") == 0.5 and store.spend("b") == 0
    dumped = json.dumps([tuple(row) for row in ledger._db.execute("SELECT * FROM shadow_runs")])
    assert "secret answer" not in dumped


def test_a_reply_with_no_comparable_answer_is_still_paid_for():
    seen = []
    provider = serve({"frontier": "(B)", "small-a": "no letter here", "small-b": "(B)"}, seen)
    _, report, _ = ask(config(answer_pattern=r"\(([A-Z])\)"), provider)
    body = report.json()
    assert body["policy_would_accept"] == 0 and body["failed_shadow_calls"] == 0
    assert body["unknown_cost_requests"] == 0 and body["projected_policy_cost_usd"] == pytest.approx(0.206)


SAME = {"frontier": "B", "small-a": "B", "small-b": "B"}


def test_requests_that_arrive_together_cannot_overspend_the_cap():
    """Forty requests at once, each shadow run costing $0.006 against a $0.01 cap. Read
    one after another they stop after two runs; arriving together they must as well."""
    seen = []
    answers, report, _ = ask(config(shadow_max_cost_usd=0.01), serve(SAME, seen, delay=0.01), count=40, together=True)
    assert all(answer.status_code == 200 for answer in answers) and seen.count("frontier") == 40
    assert seen.count("small-a") <= 2 and report.json()["shadowed_requests"] <= 2
    assert report.json()["shadow_spend_usd"] <= 0.012 + 1e-9


def test_shadow_runs_in_flight_are_bounded_and_the_rest_of_the_sample_is_skipped():
    seen = []
    holding = asyncio.Event()

    async def provider(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model != "frontier" and seen.count("frontier") > 1:
            await holding.wait()  # a cheap model that does not answer
        return httpx.Response(200, json=reply("B"))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(shadow_max_concurrent=1), ledger=Ledger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                def post(index):
                    return client.post("/v1/chat/completions", headers={"authorization": "Bearer token-a"}, json={
                        "model": "frontier", "messages": [{"role": "user", "content": f"q{index}"}]})
                await post(0)  # one finished run, so the cap has a run's cost to reserve
                waiting = asyncio.gather(*(post(index) for index in range(1, 7)))
                for _ in range(200):
                    await asyncio.sleep(0.005)
                    if seen.count("frontier") == 7:
                        break
                await asyncio.sleep(0.05)
                started = seen.count("small-a")
                holding.set()
                return started, await waiting
    started, answers = asyncio.run(scenario())
    assert started == 2 and seen.count("small-a") == 2  # the first request's run, then one held run at a time
    assert all(answer.status_code == 200 for answer in answers)


def test_shadow_sampling_needs_a_price_for_every_model_the_cap_has_to_count():
    """The cap counts known costs. A stage model or a judge with no price would make
    every shadow call free in its eyes, so sampling is refused until prices are set."""
    code = {**POLICY, "stages": [{"models": ["small-a"], "check": "tests+judge"}]}
    judge = {"tests+judge": [{"type": "judge", "model": "judge-model"}]}
    priced = {**PRICES, "anthropic:judge-model": {"input_per_million": 1.0, "output_per_million": 1.0}}
    for changes in [{"prices": {"openai:frontier": PRICES["openai:frontier"]}},
                    {"prices": {**PRICES, "openai:small-b": {"input_per_million": 1.0}}},
                    {"shadow_policies": {"a": {"openai": code}}, "checks": judge, "anthropic_api_key": "k"}]:
        with pytest.raises(ValueError, match="price"):
            create_app(config(**changes), ledger=Ledger())
    create_app(config(shadow_policies={"a": {"openai": code}}, checks=judge, anthropic_api_key="k", prices=priced),
               ledger=Ledger())
    create_app(config(prices={}, shadow_sample_rate=0.0), ledger=Ledger())  # not sampling: nothing is spent


def test_shadowing_stops_after_too_many_calls_whose_cost_is_not_known():
    seen = []
    _, report, _ = ask(config(shadow_max_cost_usd=1.0, shadow_max_unknown_cost_calls=2),
                       serve(SAME, seen, unmetered={"small-b"}), count=6)
    body = report.json()
    assert seen.count("small-b") == 3 and body["shadowed_requests"] == 3  # the third unknown cost is one too many
    assert body["unknown_cost_shadow_calls"] == 3 and body["shadow_spend_usd"] == pytest.approx(0.006)
    assert body["sampling"] == "stopped: too many shadow calls of unknown cost"
    _, report, _ = ask(config(shadow_max_cost_usd=0.01), serve(SAME, []), count=4)
    assert report.json()["sampling"] == "stopped: the spending cap is reached"
    _, report, _ = ask(config(), serve(SAME, []))
    assert report.json()["sampling"] == "on"
    _, report, _ = ask(config(), serve(SAME, []), token="token-b")
    assert report.json()["sampling"] == "no shadow policy for this tenant"


def test_the_cap_is_kept_in_memory_and_survives_a_row_that_cannot_be_written(monkeypatch):
    reads = []
    read = ShadowStore._runs
    monkeypatch.setattr(ShadowStore, "_runs", lambda self, tenant: reads.append(tenant) or read(self, tenant))
    seen = []
    ask(config(shadow_max_cost_usd=1.0), serve(SAME, seen), count=5)
    assert seen.count("small-a") == 5 and reads == ["a"]  # the table is read once, not for every request

    def broken(self, **values):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(ShadowStore, "record", broken)
    seen = []
    ask(config(shadow_max_cost_usd=0.01), serve(SAME, seen), count=4)
    assert seen.count("small-a") == 2  # the spend was real even though its row was lost


def test_shadow_store_counts_the_known_part_of_a_check_and_never_stores_a_cost_it_cannot_add():
    store = ShadowStore(Ledger())
    outcome = {"accepted": False, "stage": None, "served_by": None, "answer": None,
               "calls": [{"model": "coder", "cost_usd": float("nan"), "error": None},
                         {"model": "other", "cost_usd": 0.25, "error": None}],
               "checks": [{"check": "tests+judge", "model": "coder", "passed": False, "cost_usd": None,
                           "known_cost_usd": 0.01, "error": None}]}
    store.record(request_id="r1", tenant="a", provider="openai", policy=POLICY, outcome=outcome,
                 frontier_answer="x", frontier_cost_usd=2.0)
    report = store.report("a")
    assert store.spend("a") == pytest.approx(0.26) and report["shadow_spend_usd"] == pytest.approx(0.26)
    assert report["unknown_cost_shadow_calls"] == 2 and report["unknown_cost_requests"] == 1


def test_a_shadow_policy_without_a_reader_runs_but_says_so(caplog):
    """Shadow mode only reports, so it starts; the log and the report name the reader."""
    caplog.set_level(logging.INFO, logger="tine.proxy")
    _, report, _ = ask(config(), serve(SAME, []))
    reader = report.json()["policies"]["openai"]
    assert reader["answer_reader"] == {"answer_pattern": None, "source": "none: whole reply texts are compared"}
    assert "no answer reader" in reader["warning"] and reader["cheaper_stages"] == 1
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("shadow policy for tenant 'a' on openai: agreement stages with no answer reader" in line
               for line in warnings), warnings
    _, report, _ = ask(config(answer_pattern=r"\(([A-Z])\)"), serve(SAME, []))
    assert report.json()["policies"]["openai"]["answer_reader"] == {
        "answer_pattern": r"\(([A-Z])\)", "source": "TINE_ANSWER_PATTERN"}
    assert "warning" not in report.json()["policies"]["openai"]
    # The policy's own reader wins over the proxy's: it is part of what was frozen.
    own = {**POLICY, "answer_pattern": r"^(\w)"}
    _, report, _ = ask(config(answer_pattern=r"\(([A-Z])\)", shadow_policies={"a": {"openai": own}}), serve(SAME, []))
    assert report.json()["policies"]["openai"]["answer_reader"] == {"answer_pattern": r"^(\w)", "source": "policy file"}
    _, report, _ = ask(config(), serve(SAME, []), token="token-b")
    assert report.json()["policies"] == {}


def test_a_policy_edited_after_freezing_still_loads_and_is_marked(caplog):
    frozen = {**POLICY, "answer_pattern": r"(\w)", "provenance": {"seed": 42}}
    frozen["provenance"][DIGEST_FIELD] = policy_digest(frozen)
    _, report, _ = ask(config(shadow_policies={"a": {"openai": frozen}}), serve(SAME, []))
    assert report.json()["policies"]["openai"]["freeze_digest"] == FREEZE_UNCHANGED
    caplog.set_level(logging.INFO, logger="tine.proxy")
    edited = {**frozen, "guarantee": {**frozen["guarantee"], "epsilon": 0.2}}
    _, report, _ = ask(config(shadow_policies={"a": {"openai": edited}}), serve(SAME, []))
    assert report.json()["policies"]["openai"]["freeze_digest"] == FREEZE_EDITED
    assert report.json()["shadowed_requests"] == 1
    assert any("was edited after freezing" in record.getMessage() for record in caplog.records
               if record.levelno == logging.WARNING)


def test_a_sampled_request_with_tools_is_counted_at_the_frontier_without_shadow_calls():
    seen = []
    tools = {"tools": [{"type": "function", "function": {"name": "lookup"}}]}
    _, report, _ = ask(config(), serve(SAME, seen), count=2, extra=tools)
    assert seen == ["frontier", "frontier"]  # nothing spent on cheaper models
    body = report.json()
    assert body["shadowed_requests"] == 2 and body["sent_to_frontier_for_tools"] == 2
    assert body["policy_would_accept"] == 0 and body["shadow_calls"] == 0
    assert body["projected_policy_cost_usd"] == pytest.approx(body["frontier_cost_usd"])
    assert body["projected_savings_usd"] == pytest.approx(0.0)
    seen = []
    _, report, _ = ask(config(), serve(SAME, seen))
    assert report.json()["sent_to_frontier_for_tools"] == 0 and seen == ["frontier", "small-a", "small-b"]
    # A stopped tenant records nothing more, tools or not.
    seen = []
    _, report, _ = ask(config(shadow_max_cost_usd=0.0), serve(SAME, seen), extra=tools)
    assert report.json()["shadowed_requests"] == 0
