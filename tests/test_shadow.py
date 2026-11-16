import asyncio
import json

import httpx
import pytest

from tine.ledger import Ledger
from tine.proxy import ProxyConfig, create_app
from tine.shadow import ShadowStore, extract_answer, sampled, shadow_stages, validate_shadow_policy

POLICY = {"version": 1, "frontier": "frontier", "guarantee": {"certified": True},
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


def serve(answers, seen, fail=()):
    def provider(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model in fail:
            return httpx.Response(500, json={"error": "down"})
        return httpx.Response(200, json=reply(answers[model]))
    return provider


def ask(app_config, provider, token="token-a", count=1):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(app_config, ledger=Ledger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                headers = {"authorization": f"Bearer {token}"}
                answers = [await client.post("/v1/chat/completions", headers=headers,
                                             json={"model": "frontier", "messages": [{"role": "user", "content": f"q{i}"}]})
                           for i in range(count)]
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
                    {"shadow_answer_pattern": "("}]:
        with pytest.raises(ValueError):
            create_app(config(**changes), ledger=Ledger())
    with pytest.raises(ValueError, match="agreement stages only"):
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
    _, report, _ = ask(config(shadow_answer_pattern=r"\(([A-Z])\)"), provider)
    body = report.json()
    assert body["policy_would_accept"] == 0 and body["failed_shadow_calls"] == 0
    assert body["unknown_cost_requests"] == 0 and body["projected_policy_cost_usd"] == pytest.approx(0.206)
