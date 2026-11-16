import asyncio
import json

import httpx
import pytest

from tine.ledger import Ledger
from tine.proxy import ProxyConfig, create_app
from tine.runtime_checks import request_text, validate_checks

AGREE = {"version": 1, "frontier": "frontier", "guarantee": {"certified": True},
         "stages": [{"models": ["small-a", "small-b"], "check": "agreement"}]}
CODE = {"version": 1, "frontier": "frontier", "guarantee": {"certified": True},
        "stages": [{"models": ["coder"], "check": "tests+judge"}]}
PRICES = {**{f"openai:{name}": {"input_per_million": rate, "output_per_million": rate}
             for name, rate in (("frontier", 100.0), ("small-a", 1.0), ("small-b", 2.0), ("coder", 5.0))},
          "anthropic:judge-model": {"input_per_million": 10.0, "output_per_million": 10.0}}
CHECKS = {"tests+judge": [{"type": "webhook", "url": "https://ci.example/check"},
                          {"type": "judge", "model": "judge-model", "threshold": 0.7}]}


def config(**changes):
    values = dict(tenant_keys={"token-a": "a"}, openai_api_key="secret", anthropic_api_key="secret-a",
                  openai_frontier_model="frontier", prices=PRICES, verified_cache=False)
    values.update(changes)
    return ProxyConfig(**values)


def chat(text):
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}


def world(answers, seen, *, tests_pass=True, judge_probability=0.9, down=()):
    """One stand-in for the model provider, the customer's test webhook and the judge."""
    def handler(request):
        if request.url.host == "ci.example":
            seen.append("webhook")
            sent = json.loads(request.content)
            assert sent["tenant"] == "a" and sent["response"]["choices"] and "authorization" not in request.headers
            return httpx.Response(200, json={"passed": tests_pass, "cost_usd": 0.01})
        if request.url.path.endswith("/v1/messages"):
            seen.append("judge")
            sent = json.loads(request.content)
            assert sent["model"] == "judge-model" and "<patch>" in sent["messages"][0]["content"]
            verdict = {"verdict": "accept" if judge_probability >= 0.5 else "reject",
                       "probability": judge_probability, "reason": "r"}
            return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(verdict)}],
                                             "stop_reason": "end_turn",
                                             "usage": {"input_tokens": 500, "output_tokens": 500}})
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model in down:
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json=chat(answers[model]))
    return handler


def ask(app_config, handler, body=None):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
            ledger = Ledger()
            app = create_app(app_config, ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                headers = {"authorization": "Bearer token-a"}
                response = await client.post("/v1/chat/completions", headers=headers, json=body or {
                    "model": "frontier", "messages": [{"role": "user", "content": "fix the bug"}]})
                row = ledger.get("a", response.headers["x-tine-request-id"])
                return response, row
    return asyncio.run(scenario())


def test_agreeing_cheap_models_serve_the_answer_and_the_second_opinion_is_routing_cost():
    seen = []
    response, row = ask(config(route_policies={"a": {"openai": AGREE}}),
                        world({"small-a": "42", "small-b": " 42 ", "frontier": "42"}, seen))
    assert seen == ["small-a", "small-b"] and response.json() == chat("42")
    assert response.headers["x-tine-model"] == "small-a" and response.headers["x-tine-route"] == "stage-0"
    assert row["model"] == "small-a" and row["cost_usd"] == pytest.approx(0.002)
    assert row["routing_cost_usd"] == pytest.approx(0.004) and row["baseline_cost_usd"] is None


def test_disagreement_falls_back_to_the_frontier_and_carries_the_routing_cost():
    seen = []
    response, row = ask(config(route_policies={"a": {"openai": AGREE}}),
                        world({"small-a": "41", "small-b": "42", "frontier": "42"}, seen))
    assert seen == ["small-a", "small-b", "frontier"] and response.json() == chat("42")
    assert response.headers["x-tine-route"] == "frontier" and row["model"] == "frontier"
    assert row["cost_usd"] == pytest.approx(0.2) and row["routing_cost_usd"] == pytest.approx(0.006)
    assert row["baseline_cost_usd"] == pytest.approx(0.2)


def test_a_failing_cheap_model_or_a_streaming_request_still_gets_the_frontier():
    seen = []
    response, row = ask(config(route_policies={"a": {"openai": AGREE}}),
                        world({"small-a": "42", "frontier": "42"}, seen, down={"small-b"}))
    assert response.status_code == 200 and seen[-1] == "frontier" and row["routing_cost_usd"] is None
    seen = []
    response, _ = ask(config(route_policies={"a": {"openai": AGREE}}), world({"frontier": "42"}, seen),
                      body={"model": "frontier", "stream": True, "messages": [{"role": "user", "content": "q"}]})
    assert seen == ["frontier"] and "x-tine-route" not in response.headers


def test_the_baseline_is_estimated_only_when_asked_for():
    seen = []
    _, row = ask(config(route_policies={"a": {"openai": AGREE}}, route_estimate_baseline=True),
                 world({"small-a": "42", "small-b": "42"}, seen))
    assert row["baseline_cost_usd"] == pytest.approx(0.2)  # this answer's 2,000 tokens at the frontier's price


def test_a_code_stage_runs_the_customers_tests_then_the_judge():
    routed = config(route_policies={"a": {"openai": CODE}}, checks=CHECKS)
    seen = []
    response, row = ask(routed, world({"coder": "diff --git a b", "frontier": "frontier patch"}, seen))
    assert seen == ["coder", "webhook", "judge"] and response.headers["x-tine-model"] == "coder"
    # coder $0.01; webhook $0.01 and judge 1,000 tokens at $10 per million = $0.01 are routing cost.
    assert row["cost_usd"] == pytest.approx(0.01) and row["routing_cost_usd"] == pytest.approx(0.02)
    seen = []
    response, row = ask(routed, world({"coder": "bad", "frontier": "frontier patch"}, seen, tests_pass=False))
    assert seen == ["coder", "webhook", "frontier"]  # the judge is not paid when the tests already failed
    assert response.json() == chat("frontier patch") and row["routing_cost_usd"] == pytest.approx(0.02)
    seen = []
    response, _ = ask(routed, world({"coder": "x", "frontier": "y"}, seen, judge_probability=0.6))
    assert seen == ["coder", "webhook", "judge", "frontier"]  # accepted by verdict but below the 0.7 threshold


def test_shadow_mode_runs_named_checks_without_comparing_answers():
    shadowed = config(shadow_policies={"a": {"openai": CODE}}, shadow_sample_rate=1.0, shadow_max_cost_usd=1.0,
                      checks=CHECKS)
    seen = []

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                world({"coder": "cheap patch", "frontier": "frontier patch"}, seen))) as upstream:
            app = create_app(shadowed, ledger=Ledger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                headers = {"authorization": "Bearer token-a"}
                answer = await client.post("/v1/chat/completions", headers=headers,
                                           json={"model": "frontier", "messages": [{"role": "user", "content": "q"}]})
                return answer, (await client.get("/v1/shadow", headers=headers)).json()
    answer, report = asyncio.run(scenario())
    assert answer.json() == chat("frontier patch") and seen == ["frontier", "coder", "webhook", "judge"]
    assert report["policy_would_accept"] == 1 and report["accepted_compared_with_frontier"] == 0
    assert report["accepted_matching_frontier_share"] is None
    assert report["projected_policy_cost_usd"] == pytest.approx(0.03) and report["shadow_spend_usd"] == pytest.approx(0.03)


def test_policies_and_checks_are_validated_before_anything_runs():
    for changes in [{"route_policies": {"a": {"openai": CODE}}},                       # no check configured
                    {"route_policies": {"a": {"openai": AGREE}}, "shadow_policies": {"a": {"openai": AGREE}}},
                    {"route_policies": {"nobody": {"openai": AGREE}}},
                    {"checks": {"agreement": [{"type": "webhook", "url": "https://x.example"}]}},
                    {"checks": {"t": [{"type": "webhook", "url": "ftp://x.example"}]}},
                    {"checks": {"t": [{"type": "webhook", "url": "https://user:pw@x.example"}]}},
                    {"checks": {"t": [{"type": "judge"}]}},
                    {"checks": {"t": [{"type": "judge", "model": "m", "threshold": 2}]}},
                    {"checks": {"t": [{"type": "judge", "model": "m"}]}, "anthropic_api_key": ""},
                    {"checks": {"t": []}}, {"route_estimate_baseline": "yes"}]:
        with pytest.raises(ValueError):
            create_app(config(**changes), ledger=Ledger())
    assert validate_checks(CHECKS) is CHECKS
    assert request_text({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "a"},
                                      {"role": "user", "content": [{"type": "text", "text": "b"}]}]}) == "a\n\nb"
