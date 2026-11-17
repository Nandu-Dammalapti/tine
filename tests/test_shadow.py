import asyncio
import json
import logging
from dataclasses import replace

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


def test_a_shadow_run_that_cost_nothing_does_not_lift_the_one_run_at_a_time_limit(tmp_path):
    """A run with no model call (a judged stage skipped for a long request, a request
    skipped for tools) costs nothing. Reserving that zero for every run in flight would
    let them all through, however much they cost."""
    def admitted(store, times=5):
        return [store.admit("a", 1.0, 100) for _ in range(times)]
    live = ShadowStore(Ledger())
    assert admitted(live) == [True, False, False, False, False]
    live.settle("a", [{"model": "check:judge", "cost_usd": 0.0, "error": None, "reason": "request_too_long"}])
    assert admitted(live) == [True, False, False, False, False]
    live.settle("a", [{"model": "small-a", "cost_usd": 0.25, "error": None}])
    assert admitted(live) == [True, True, True, False, False]  # $0.25 reserved for each run in flight
    path = tmp_path / "ledger.sqlite3"
    skipped = ShadowStore(Ledger(path))
    skipped.record(request_id="r1", tenant="a", provider="openai", policy=POLICY, outcome={
        "accepted": False, "stage": None, "served_by": None, "answer": None, "calls": [], "checks": []},
        frontier_answer="A", frontier_cost_usd=1.0, skip_reason="tools")
    skipped._ledger.close()
    restarted = ShadowStore(Ledger(path))
    assert restarted._budget("a")["largest"] is None
    assert admitted(restarted) == [True, False, False, False, False]


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
    # Three of six calls is above the default share of 20% too; the reason names the failures.
    assert body["sampling"] == ("stopped: too many shadow calls of unknown cost: 3 of 6 calls and checks (50.0%), "
                                "more than 2 and more than 20.0% of them. Most common: small-b: no usage or no "
                                "price in the reply (3)")
    assert body["skipped_stopped"] == 3 and body["sampled_requests"] == 6 and body["shadow_coverage"] == 0.5
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
    _, report, _ = ask(config(shadow_max_cost_usd=0.01), serve(SAME, seen), count=4)
    assert seen.count("small-a") == 2  # the spend was real even though its row was lost
    body = report.json()  # and the report says the runs failed, rather than leaving them out
    assert (body["shadowed_requests"], body["failed_shadow_runs"], body["skipped_stopped"]) == (0, 2, 2)
    assert body["sampled_requests"] == 4 and body["shadow_coverage"] == 0.0


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


BOXED = r"\\boxed\{(.+)\}"


@pytest.mark.parametrize("mode", ["exact", "math"])
def test_shadow_agreement_and_the_frontier_comparison_use_the_policys_answer_comparison(mode, caplog):
    """FinQA-style replies: the same value written three ways. Exact comparison sees
    three answers; math comparison sees one."""
    caplog.set_level(logging.INFO, logger="tine.proxy")
    policy = {**POLICY, "answer_pattern": BOXED, **({"answer_compare": mode} if mode == "math" else {})}
    replies = {"frontier": r"So \boxed{14.5}", "small-a": r"The answer is \boxed{14.5\%}", "small-b": r"\boxed{14.50}"}
    _, report, _ = ask(config(shadow_policies={"a": {"openai": policy}}), serve(replies, []))
    body = report.json()
    assert body["policies"]["openai"]["answer_compare"] == mode
    assert any(f"answers compared {mode}" in record.getMessage() for record in caplog.records)
    if mode == "exact":
        assert body["policy_would_accept"] == 0 and body["accepted_compared_with_frontier"] == 0
    else:
        assert body["policy_would_accept"] == 1 and body["accepted_matching_frontier"] == 1
    # By value, an accepted answer that is a different number still differs from the frontier's.
    replies = {**replies, "frontier": r"\boxed{15}"}
    _, report, _ = ask(config(shadow_policies={"a": {"openai": policy}}), serve(replies, []))
    if mode == "math":
        body = report.json()
        assert body["policy_would_accept"] == 1 and body["accepted_compared_with_frontier"] == 1
        assert body["accepted_matching_frontier"] == 0


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
    # It is still counted, so the coverage figures say what was not shadowed.
    assert report.json()["skipped_stopped"] == 1 and report.json()["sampled_requests"] == 1


def flaky(fail_on, status=503):
    """Every model answers "B", except that small-b fails with ``status`` on the requests
    whose prompt is in ``fail_on``: a provider or CI that fails now and then."""
    async def provider(request):
        sent = json.loads(request.content)
        if sent["model"] == "small-b" and sent["messages"][0]["content"] in fail_on:
            return httpx.Response(status, json={"error": "flaky"})
        return httpx.Response(200, json=reply("B"))
    return provider


def test_occasional_failures_of_unknown_cost_do_not_stop_shadowing():
    """Three failures in twenty requests (40 calls) are more than two, but under 20% of
    the calls: shadowing goes on. On the count alone it stops, and says why."""
    _, report, _ = ask(config(shadow_max_unknown_cost_calls=2), flaky({"q5", "q10", "q15"}), count=20)
    body = report.json()
    assert body["shadowed_requests"] == 20 and body["unknown_cost_shadow_calls"] == 3 and body["sampling"] == "on"
    assert body["failed_shadow_calls"] == 3 and body["shadow_coverage"] == 1.0
    _, report, _ = ask(config(shadow_max_unknown_cost_calls=2, shadow_max_unknown_cost_share=0.0),
                       flaky({"q5", "q10", "q15"}), count=20)
    body = report.json()
    assert body["shadowed_requests"] == 16 and body["skipped_stopped"] == 4
    assert body["sampling"].startswith("stopped: too many shadow calls of unknown cost: 3 of 32 calls and checks")
    assert body["sampling"].endswith("Most common: small-b: HTTP 503 (3)")  # the failures are named
    # Failing on most requests still stops it, with both limits in force.
    _, report, _ = ask(config(shadow_max_unknown_cost_calls=2), flaky({f"q{i}" for i in range(20)}, 404), count=20)
    body = report.json()
    assert body["shadowed_requests"] == 3 and body["sampling"].endswith("small-b: HTTP 404 (3)")


def test_the_unknown_cost_stop_is_read_back_from_the_rows_after_a_restart():
    ledger = Ledger()
    failed = {"accepted": False, "stage": None, "served_by": None, "answer": None,
              "calls": [{"model": "small-a", "cost_usd": 0.1, "error": None},
                        {"model": "small-b", "cost_usd": None, "error": "HTTPStatusError", "status": 429}]}
    ShadowStore(ledger).record(request_id="r1", tenant="a", provider="openai", policy=POLICY, outcome=failed,
                               frontier_answer="B", frontier_cost_usd=1.0)
    restarted = ShadowStore(ledger)
    assert restarted.state("a", 5.0, 0, 0.2) == ("stopped: too many shadow calls of unknown cost: 1 of 2 calls and "
                                                 "checks (50.0%), more than 0 and more than 20.0% of them. Most "
                                                 "common: small-b: HTTP 429 (1)")
    assert restarted.state("a", 5.0, 0, 0.5) == "on" and restarted.state("a", 5.0, 1, 0.2) == "on"
    assert restarted.admit("a", 5.0, 1, 0.2) and not restarted.admit("b", 0.0, 1, 0.2)


def hold_cheap_models(app_config, posts):
    """Send ``posts`` (token, prompt) at once after one finished request, with the cheap
    models held until every frontier answer is back; return both tenants' reports."""
    seen = []
    holding = asyncio.Event()

    async def provider(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model != "frontier" and seen.count("frontier") > 1:
            await holding.wait()
        return httpx.Response(200, json=reply("B"))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(app_config, ledger=Ledger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                def post(token, prompt):
                    return client.post("/v1/chat/completions", headers={"authorization": f"Bearer {token}"}, json={
                        "model": "frontier", "messages": [{"role": "user", "content": prompt}]})
                await post("token-a", "first")  # one finished run, so the cap has a run's cost to reserve
                waiting = asyncio.gather(*(post(token, prompt) for token, prompt in posts))
                for _ in range(200):
                    await asyncio.sleep(0.005)
                    if seen.count("frontier") == len(posts) + 1:
                        break
                await asyncio.sleep(0.05)
                holding.set()
                await waiting
                await asyncio.sleep(0.05)
                return {token: (await client.get("/v1/shadow", headers={"authorization": f"Bearer {token}"})).json()
                        for token in ("token-a", "token-b")}
    return asyncio.run(scenario())


def test_sampled_requests_skipped_while_runs_are_busy_are_counted():
    reports = hold_cheap_models(config(shadow_max_concurrent=1), [("token-a", f"q{i}") for i in range(6)])
    body = reports["token-a"]
    assert body["shadowed_requests"] == 2 and body["skipped_busy"] == 5 and body["sampled_requests"] == 7
    assert body["shadow_coverage"] == pytest.approx(2 / 7) and body["skipped_stopped"] == 0


def test_a_limit_per_tenant_keeps_one_busy_tenant_from_taking_every_slot():
    both = config(shadow_policies={"a": {"openai": POLICY}, "b": {"openai": {**POLICY, "tenant": "b"}}},
                  shadow_max_concurrent=3)
    posts = [("token-a", f"a{i}") for i in range(4)] + [("token-b", "b0")]
    shared = hold_cheap_models(both, posts)
    # Without a limit of its own, tenant a fills the proxy's slots and b's sample is lost.
    assert shared["token-b"]["shadowed_requests"] == 0 and shared["token-b"]["skipped_busy"] == 1
    capped = hold_cheap_models(replace(both, shadow_max_concurrent_per_tenant=1), posts)
    assert capped["token-b"]["shadowed_requests"] == 1 and capped["token-b"]["shadow_coverage"] == 1.0
    assert capped["token-a"]["shadowed_requests"] == 2 and capped["token-a"]["skipped_busy"] == 3
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="per_tenant"):
            create_app(config(shadow_max_concurrent_per_tenant=bad), ledger=Ledger())


def test_a_sampled_request_that_streamed_is_counted_and_not_shadowed():
    seen = []
    # One stream carries usage and one does not: a known cost changes none of the counts.
    streams = [b'data: {"usage": null}\n\ndata: {"choices": [], "usage": {"prompt_tokens": 1000, '
               b'"completion_tokens": 1000}}\n\ndata: [DONE]\n\n', b"data: {}\n\ndata: [DONE]\n\n"]

    async def provider(request):
        sent = json.loads(request.content)
        seen.append(sent["model"])
        if sent.get("stream"):
            return httpx.Response(200, content=streams.pop(0), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=reply("B"))

    async def scenario():
        ledger = Ledger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(), ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                headers = {"authorization": "Bearer token-a"}
                costs = []
                for stream in (True, True, False):
                    response = await client.post("/v1/chat/completions", headers=headers, json={
                        "model": "frontier", "stream": stream, "messages": [{"role": "user", "content": "q"}]})
                    assert response.status_code == 200
                    costs.append(ledger.get("a", response.headers["x-tine-request-id"])["cost_usd"])
                assert costs == [0.2, None, 0.2]
                return (await client.get("/v1/shadow", headers=headers)).json()
    body = asyncio.run(scenario())
    assert seen == ["frontier", "frontier", "frontier", "small-a", "small-b"]
    assert body["streamed_not_shadowed"] == 2 and body["sampled_requests"] == 1 and body["shadow_coverage"] == 1.0
    assert body["shadowed_requests"] == 1


def test_the_counts_are_unknown_for_a_ledger_that_has_none():
    ledger = Ledger()
    store = ShadowStore(ledger)
    assert store.report("a")["sampled_requests"] == 0 and store.report("a")["shadow_coverage"] is None
    with pytest.raises(ValueError):
        store.count("a", "shadowed")  # a shadowed request is counted by its row
    ledger._db.execute("DROP TABLE shadow_counts")
    report = store.report("a")
    assert report["sampled_requests"] is None and report["streamed_not_shadowed"] is None


def shadow_rows(store, matching, differing, not_accepted=0):
    outcome = {"accepted": True, "stage": 0, "served_by": "small-a", "answer": "A",
               "calls": [{"model": "small-a", "cost_usd": 0.01, "error": None}]}
    rows = [("A", outcome)] * matching + [("B", outcome)] * differing
    rows += [("A", {**outcome, "accepted": False, "stage": None, "served_by": None})] * not_accepted
    for index, (frontier, run) in enumerate(rows):
        store.record(request_id=f"r{index}", tenant="a", provider="openai", policy=POLICY, outcome=run,
                     frontier_answer=frontier, frontier_cost_usd=1.0)


def test_too_many_answers_that_differ_from_the_frontier_cannot_confirm_the_certification():
    store = ShadowStore(Ledger())
    shadow_rows(store, matching=18, differing=12, not_accepted=10)  # cheap models agreeing on wrong answers
    report = store.report("a", epsilon=0.02)
    assert report["accepted_differing_from_frontier"] == 12 and report["accepted_differing_from_frontier_share"] == 0.4
    assert report["drift_check"] == "warning" and report["drift_warning"].startswith(
        "Cannot confirm the certification on this traffic: 12 of 30 accepted cheaper answers differ from the "
        "frontier model's answer, 30.00% of the 40 shadowed requests whose answer was compared with the frontier's "
        "or sent to the frontier, more than the policy's quality limit of 2.00%.")
    assert (report["drift_requests"], report["drift_share"]) == (40, 0.3)
    assert store.report("a")["drift_warning"] is None and store.report("a")["drift_check"].startswith("not checked:")
    assert store.report("a", epsilon=0.5)["drift_check"].startswith("within the limit: 12 of 30")
    few = ShadowStore(Ledger())
    shadow_rows(few, matching=10, differing=19)
    assert few.report("a", epsilon=0.02)["drift_warning"] is None
    assert few.report("a", epsilon=0.02)["drift_check"] == ("not checked yet: 29 accepted answers compared with the "
                                                            "frontier's, 30 needed")
    assert few.report("a", epsilon=0.02, min_compared=29)["drift_check"] == "warning"


def test_differing_answers_are_counted_over_all_the_traffic_the_limit_is_about():
    """Epsilon bounds the loss over all tasks, and a request sent to the frontier loses
    nothing. 4 differing answers among 50 accepted of 100 shadowed requests are at most
    4% of the traffic, within a 5% limit, though they are 8% of the accepted answers."""
    store = ShadowStore(Ledger())
    shadow_rows(store, matching=46, differing=4, not_accepted=50)
    report = store.report("a", epsilon=0.05)
    assert report["accepted_differing_from_frontier_share"] == 0.08 and report["drift_share"] == 0.04
    assert report["drift_warning"] is None and report["drift_check"] == (
        "within the limit: 4 of 50 accepted answers differ from the frontier's, 4.00% of the 100 shadowed requests "
        "whose answer was compared with the frontier's or sent to the frontier, at most the policy's quality limit "
        "of 5.00%")
    assert store.report("a", epsilon=0.03)["drift_check"] == "warning"
    # Accepted answers that were not compared (a code check) cannot be told either way: left out.
    for index in range(20):
        store.record(request_id=f"code{index}", tenant="a", provider="openai", policy=POLICY, frontier_answer="A",
                     frontier_cost_usd=1.0, outcome={"accepted": True, "comparable": False, "stage": 0,
                                                     "served_by": "small-a", "answer": "x", "calls": []})
    assert store.report("a", epsilon=0.05)["drift_requests"] == 100


def test_the_shadow_endpoint_checks_differing_answers_against_the_policys_limit():
    limited = {**POLICY, "guarantee": {**POLICY["guarantee"], "epsilon": 0.02, "bound": 0.01}}
    agree_wrongly = serve({"frontier": "B", "small-a": "A", "small-b": "A"}, [])
    _, report, _ = ask(config(shadow_policies={"a": {"openai": limited}}), agree_wrongly, count=30)
    body = report.json()
    assert body["drift_epsilon"] == 0.02 and body["drift_warning"].startswith(
        "Cannot confirm the certification on this traffic: 30 of 30 accepted cheaper answers")
    _, report, _ = ask(config(shadow_policies={"a": {"openai": limited}}), serve(SAME, []), count=30)
    assert report.json()["drift_warning"] is None and report.json()["drift_check"].startswith("within the limit")
    # A policy with stages must state its limit (validate_policy), so the endpoint always
    # has one to check against; each tenant's own policy gives it.
    _, report, _ = ask(config(), agree_wrongly, count=30)
    assert report.json()["drift_epsilon"] == POLICY["guarantee"]["epsilon"] and report.json()["drift_warning"]


def test_the_new_shadow_settings_are_read_from_the_environment_and_checked(monkeypatch):
    monkeypatch.setenv("TINE_TENANT_KEYS", '{"token-a": "a"}')
    for name in ("TINE_SHADOW_POLICIES", "TINE_ROUTE_POLICIES", "TINE_CHECKS", "TINE_TENANT_CHECKS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TINE_SHADOW_MAX_CONCURRENT_PER_TENANT", "2")
    monkeypatch.setenv("TINE_SHADOW_MAX_UNKNOWN_COST_SHARE", "0.35")
    loaded = ProxyConfig.from_env()
    loaded.validate()
    assert (loaded.shadow_max_concurrent_per_tenant, loaded.shadow_max_unknown_cost_share) == (2, 0.35)
    monkeypatch.delenv("TINE_SHADOW_MAX_CONCURRENT_PER_TENANT")
    monkeypatch.delenv("TINE_SHADOW_MAX_UNKNOWN_COST_SHARE")
    defaults = ProxyConfig.from_env()
    assert (defaults.shadow_max_concurrent_per_tenant, defaults.shadow_max_unknown_cost_share) == (None, 0.2)
    for bad in (-0.1, 1.5, float("nan"), True):
        with pytest.raises(ValueError, match="unknown_cost_share"):
            create_app(config(shadow_max_unknown_cost_share=bad), ledger=Ledger())
