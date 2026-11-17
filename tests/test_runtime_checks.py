import asyncio
import json

import httpx
import pytest

from tine.runtime_checks import build_checks
from tine.shadow import shadow_stages


def check_reply(*, stop_reason="end_turn", tool_type=None, threshold=None, cost=0.03):
    async def scenario():
        text = json.dumps({"verdict": "accept", "probability": 0.9, "reason": "looks correct"})
        content = [{"type": "text", "text": text}]
        if tool_type:
            content.append({"type": tool_type, "id": "tool-1", "name": "review", "input": {}})
        payload = {"content": content}
        if stop_reason is not None:
            payload["stop_reason"] = stop_reason
        calls = []

        def judge(request):
            calls.append(request)
            return httpx.Response(200, json=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(judge)) as upstream:
            steps = [{"type": "judge", "model": "judge", "threshold": threshold}]
            checks = build_checks(
                {"review": steps}, upstream=upstream, tenant="tenant", provider="openai",
                body={"messages": [{"role": "user", "content": "Fix this bug"}]},
                anthropic_url="https://judge.example/v1/messages", anthropic_key="test-key",
                cost_of=lambda *args: cost,
            )
            policy = {"version": 1, "tenant": "tenant", "frontier": "frontier",
                      "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
                      "stages": [{"models": ["cheap"], "check": "review"}]}

            async def candidate(model):
                return {"model": model, "answer": "patch", "cost_usd": 0.01,
                        "payload": {"choices": [{"message": {"content": "patch"}, "finish_reason": "stop"}]}}

            outcome = await shadow_stages(policy, candidate, checks)
            assert len(calls) == 1
            return outcome

    return asyncio.run(scenario())


@pytest.mark.parametrize("stop_reason", ["max_tokens", "pause_turn", "refusal", "tool_use", "unknown", None])
@pytest.mark.parametrize("threshold", [None, 0.8])
def test_unfinished_judge_approval_rejects_candidate_and_keeps_cost(stop_reason, threshold):
    outcome = check_reply(stop_reason=stop_reason, threshold=threshold)
    assert outcome["accepted"] is False
    check = outcome["checks"][0]
    assert check["cost_usd"] == 0.03
    assert check["steps"] == [{"type": "judge", "passed": False, "cost_usd": 0.03,
                               "reason": "not_a_finished_judge_reply"}]


@pytest.mark.parametrize("tool_type", ["tool_use", "server_tool_use"])
def test_judge_tool_call_cannot_be_used_as_a_final_approval(tool_type):
    outcome = check_reply(tool_type=tool_type)
    assert outcome["accepted"] is False
    assert outcome["checks"][0]["cost_usd"] == 0.03
    assert outcome["checks"][0]["steps"] == [{"type": "judge", "passed": False, "cost_usd": 0.03,
                                              "reason": "not_a_finished_judge_reply"}]


@pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence"])
@pytest.mark.parametrize("cost", [0.03, None])
def test_completed_judge_approval_retains_acceptance_and_known_or_unknown_cost(stop_reason, cost):
    outcome = check_reply(stop_reason=stop_reason, threshold=0.8, cost=cost)
    assert outcome["accepted"] is True
    assert outcome["checks"][0]["cost_usd"] == cost
