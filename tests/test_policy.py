import json

import numpy as np
import pandas as pd
import pytest

from tine.ledger import Ledger
from tine.policy import dumps, freeze_policy, run_policy, same_answer, validate_policy


def recorded(tasks=600, seed=0):
    """A frontier that is right 90% of the time and a cheap pair that agrees on 70% of
    tasks and is always right when it agrees."""
    rng = np.random.default_rng(seed)
    rows, checks = [], []
    for index in range(tasks):
        task_id = f"t{index}"
        agree = rng.random() < 0.7
        base = {"tenant": "acme", "task_id": task_id, "prompt": f"question {index}", "task_type": "quiz"}
        rows.append({**base, "model": "frontier", "success": bool(rng.random() < 0.9), "cost_usd": 10.0})
        rows.append({**base, "model": "a+b", "success": agree, "cost_usd": 1.0})
        checks.append({"task_id": task_id, "model": "a+b", "passed": agree})
    return pd.DataFrame(rows), pd.DataFrame(checks)


def test_freeze_policy_certifies_an_agreement_stage_and_reports_held_out_savings():
    frame, checks = recorded()
    policy = freeze_policy(frame, "frontier", checks, epsilon=0.02)
    assert policy["stages"] == [{"models": ["a", "b"], "check": "agreement"}]
    assert policy["guarantee"]["certified"] and policy["guarantee"]["bound"] <= 0.02
    assert policy["tenant"] == "acme" and policy["task_type"] == "quiz"
    # About 70% of requests stop at the pair (cost 1); the rest also pay the frontier.
    assert 0.5 < policy["estimate"]["savings"] < 0.7
    assert policy["estimate"]["accuracy"] >= policy["estimate"]["frontier_accuracy"]
    assert json.loads(dumps(policy))["version"] == 1


def test_freeze_policy_falls_back_to_the_frontier_when_nothing_certifies():
    frame, checks = recorded()
    frame.loc[frame.model == "a+b", "success"] = False  # the pair agrees on wrong answers
    policy = freeze_policy(frame, "frontier", checks, epsilon=0.02)
    assert policy["stages"] == [] and not policy["guarantee"]["certified"]
    assert policy["estimate"]["savings"] == 0 and validate_policy(policy) is policy


POLICY = {"version": 1, "frontier": "big", "guarantee": {"certified": True},
          "stages": [{"models": ["a", "b"], "check": "agreement"}, {"models": ["c"], "check": "none"}]}


def caller(answers, costs=None, fail=()):
    made = []

    def call(model):
        made.append(model)
        if model in fail:
            raise TimeoutError(model)
        return {"answer": answers[model], "cost_usd": (costs or {}).get(model, 1.0), "model": model}
    return call, made


def test_run_policy_returns_the_agreed_answer_and_counts_the_second_opinion_as_routing():
    call, made = caller({"a": "  42 ", "b": "42"}, {"a": 1.0, "b": 2.0})
    outcome = run_policy(POLICY, call)
    assert made == ["a", "b"] and outcome["served_by"] == "a" and outcome["stage"] == 0
    assert outcome["answer_cost_usd"] == 1.0 and outcome["routing_cost_usd"] == 2.0


def test_run_policy_escalates_on_disagreement_and_on_a_failed_cheap_call():
    call, made = caller({"a": "1", "b": "2", "c": "3"})
    outcome = run_policy(POLICY, call)
    assert made == ["a", "b", "c"] and outcome["served_by"] == "c" and outcome["routing_cost_usd"] == 2.0
    call, made = caller({"a": "1", "big": "9"}, {"a": 1.0, "big": 10.0}, fail={"b", "c"})
    outcome = run_policy(POLICY, call)
    assert made == ["a", "b", "c", "big"] and outcome["stage"] == "frontier"
    assert outcome["answer_cost_usd"] == 10.0 and outcome["routing_cost_usd"] == 1.0
    assert outcome["unknown_cost_calls"] == 2 and outcome["calls"][1]["error"] == "TimeoutError"


def test_a_frontier_failure_is_raised_and_missing_answers_never_agree():
    call, _ = caller({"a": None, "b": None, "c": "x"}, fail={"c", "big"})
    with pytest.raises(TimeoutError):
        run_policy(POLICY, call)
    assert not same_answer([None, None]) and not same_answer(["", ""]) and same_answer(["a b", "a  b"])


def test_validate_policy_rejects_what_it_cannot_run_faithfully():
    for broken in [{**POLICY, "version": 2}, {**POLICY, "stages": [{"models": ["a"], "check": "judge"}]},
                   {**POLICY, "stages": [{"models": ["a", "a"], "check": "agreement"}]},
                   {**POLICY, "stages": [{"models": ["big", "a"], "check": "agreement"}]},
                   {**POLICY, "guarantee": {"certified": False}}]:
        with pytest.raises(ValueError):
            validate_policy(broken)


def test_ledger_reports_savings_net_of_routing_cost():
    ledger = Ledger()
    body = {"model": "a", "messages": [{"role": "user", "content": "hi"}]}
    ledger.record(tenant="acme", provider="openai", model="a", request_body=body, success=True,
                  cost_usd=1.0, baseline_cost_usd=10.0, routing_cost_usd=2.0)
    ledger.record(tenant="acme", provider="openai", model="big", request_body={**body, "model": "big"},
                  success=True, cost_usd=10.0, baseline_cost_usd=10.0)
    report = ledger.report("acme")
    assert report["verified_routing_cost_usd"] == 2.0 and report["verified_savings_usd"] == 7.0
    assert report["savings_fraction"] == pytest.approx(1 - 13 / 20)
    with pytest.raises(ValueError):
        ledger.record(tenant="acme", provider="openai", model="a", request_body=body, routing_cost_usd=-1)


def test_a_model_in_two_stages_is_called_once_per_request():
    policy = {**POLICY, "stages": [{"models": ["a", "b"], "check": "agreement"},
                                   {"models": ["b", "c"], "check": "agreement"}]}
    call, made = caller({"a": "1", "b": "2", "c": "2"})
    outcome = run_policy(policy, call)
    assert made == ["a", "b", "c"] and outcome["served_by"] == "b" and outcome["stage"] == 1
    assert outcome["answer_cost_usd"] == 1.0 and outcome["routing_cost_usd"] == 2.0
