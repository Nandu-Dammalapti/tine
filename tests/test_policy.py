import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from tine.__main__ import main
from tine.data import write_outcomes
from tine.ledger import Ledger
from tine.policy import (
    DIGEST_FIELD, FREEZE_EDITED, FREEZE_NO_DIGEST, FREEZE_UNCHANGED, dumps, freeze_policy, freeze_state, policy_digest,
    run_policy, run_stages_async, same_answer, validate_policy,
)


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


POLICY = {"version": 1, "frontier": "big", "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
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
    # Two cheaper calls failed before their cost was known, so the routing cost is not
    # known either. The part that is known is reported under its own name.
    assert outcome["answer_cost_usd"] == 10.0 and outcome["routing_cost_usd"] is None
    assert outcome["known_routing_cost_usd"] == 1.0
    assert outcome["unknown_cost_calls"] == 2 and outcome["calls"][1]["error"] == "TimeoutError"
    assert "status" not in outcome["calls"][1]  # no HTTP reply, so no status


def test_a_failed_call_records_the_http_status_it_was_refused_with():
    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")

    def call(model):
        if model == "b":
            httpx.Response(404, request=request).raise_for_status()
        return {"answer": "1", "cost_usd": 1.0}
    outcome = run_policy({**POLICY, "stages": POLICY["stages"][:1]}, call)
    assert outcome["calls"][1] == {"model": "b", "cost_usd": None, "error": "HTTPStatusError", "status": 404}


def test_a_frontier_failure_is_raised_and_missing_answers_never_agree():
    call, _ = caller({"a": None, "b": None, "c": "x"}, fail={"c", "big"})
    with pytest.raises(TimeoutError):
        run_policy(POLICY, call)
    assert not same_answer([None, None]) and not same_answer(["", ""]) and same_answer(["a b", "a  b"])


def test_validate_policy_rejects_what_it_cannot_run_faithfully():
    for broken in [{**POLICY, "version": 2}, {**POLICY, "stages": [{"models": ["a", "b"], "check": "judge"}]},
                   {**POLICY, "stages": [{"models": ["a", "a"], "check": "agreement"}]},
                   {**POLICY, "stages": [{"models": ["big", "a"], "check": "agreement"}]},
                   {**POLICY, "guarantee": {"certified": False}},
                   {**POLICY, "version": True},
                   {**POLICY, "stages": [{"models": ["a", "a", "b"], "check": "agreement"}]},
                   {**POLICY, "guarantee": {"certified": True, "epsilon": 0.04, "bound": 0.9}}]:
        with pytest.raises(ValueError):
            validate_policy(broken)
    validate_policy({**POLICY, "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.035}})


def test_ledger_reports_savings_net_of_routing_cost():
    ledger = Ledger()
    body = {"model": "a", "messages": [{"role": "user", "content": "hi"}]}
    ledger.record(tenant="acme", provider="openai", model="a", request_body=body, success=True,
                  cost_usd=1.0, baseline_cost_usd=10.0, routing_cost_usd=2.0)
    ledger.record(tenant="acme", provider="openai", model="big", request_body={**body, "model": "big"},
                  success=True, cost_usd=10.0, baseline_cost_usd=10.0, routing_cost_usd=0.0)
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


JUDGED = {"version": 1, "frontier": "big", "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
          "stages": [{"models": ["a"], "check": "tests+judge"}, {"models": ["b"], "check": "tests+judge"}]}


def test_a_named_check_decides_a_stage_and_its_cost_is_routing_cost():
    call, made = caller({"a": "patch-a", "b": "patch-b"}, {"a": 1.0, "b": 2.0, "big": 10.0})
    judged = []

    def judge(result):
        judged.append(result["model"])
        return {"passed": result["model"] == "b", "cost_usd": 0.5}
    outcome = run_policy(JUDGED, call, checks={"tests+judge": judge})
    assert made == ["a", "b"] and judged == ["a", "b"] and outcome["served_by"] == "b"
    assert outcome["answer_cost_usd"] == 2.0 and outcome["routing_cost_usd"] == pytest.approx(1.0 + 0.5 + 0.5)
    assert [entry["passed"] for entry in outcome["checks"]] == [False, True]


def test_a_failing_check_escalates_and_a_missing_check_runs_nothing():
    call, made = caller({"a": "x", "b": "y", "big": "z"}, {"a": 1.0, "b": 2.0, "big": 10.0})

    def broken(result):
        raise RuntimeError("judge down")
    outcome = run_policy(JUDGED, call, checks={"tests+judge": broken})
    assert outcome["stage"] == "frontier" and outcome["routing_cost_usd"] is None
    assert outcome["known_routing_cost_usd"] == 3.0
    assert outcome["checks"][0]["error"] == "RuntimeError" and outcome["unknown_cost_calls"] == 2
    call, made = caller({"a": "x"})
    with pytest.raises(ValueError, match="No check supplied"):
        run_policy(JUDGED, call)
    assert made == []


def test_freeze_policy_names_the_measured_check_on_single_model_stages():
    frame, checks = recorded()
    frame["model"] = frame.model.replace({"a+b": "cheap"})
    checks["model"] = "cheap"
    policy = freeze_policy(frame, "frontier", checks, epsilon=0.02, single_check="tests+judge")
    assert policy["stages"] == [{"models": ["cheap"], "check": "tests+judge"}]
    with pytest.raises(ValueError):
        freeze_policy(frame, "frontier", checks, epsilon=0.02, single_check="agreement")


def test_an_unknown_cost_makes_the_routing_cost_unknown_not_zero():
    """A cheaper call with no known cost and a check that reports none: the request must
    not reach the ledger as if routing had been free."""
    policy = {"version": 1, "frontier": "big", "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
              "stages": [{"models": ["a"], "check": "judge"}]}

    def call(model):
        return {"answer": "x", "cost_usd": 10.0 if model == "big" else None}
    outcome = run_policy(policy, call, checks={"judge": lambda result: {"passed": False}})
    assert outcome["answer_cost_usd"] == 10.0 and outcome["unknown_cost_calls"] == 2
    assert outcome["routing_cost_usd"] is None and outcome["known_routing_cost_usd"] == 0.0
    ledger = Ledger()
    ledger.record(tenant="acme", provider="openai", model="big", request_body={"q": 1}, success=True,
                  cost_usd=outcome["answer_cost_usd"], baseline_cost_usd=10.0,
                  routing_cost_usd=outcome["routing_cost_usd"])
    report = ledger.report("acme")
    assert report["unknown_routing_cost_requests"] == 1 and report["comparable_verified_requests"] == 0


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1.0, 10**400, 1e300, True, "0.5"])
def test_a_check_cost_that_is_not_a_usable_amount_is_unknown(cost):
    policy = {"version": 1, "frontier": "big", "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
              "stages": [{"models": ["a"], "check": "judge"}]}
    call, _ = caller({"a": "x", "big": "y"})
    outcome = run_policy(policy, call, checks={"judge": lambda result: {"passed": True, "cost_usd": cost}})
    assert outcome["served_by"] == "a" and outcome["checks"][0]["cost_usd"] is None
    assert outcome["routing_cost_usd"] is None and outcome["known_routing_cost_usd"] == 0.0


def test_missing_answer_words_never_agree():
    for word in ("None", "null", "[None]", " None "):
        assert not same_answer([word, word])
    assert same_answer(["none of the above", "none of the  above"])


def test_stages_stop_at_the_deadline_and_keep_what_was_already_spent():
    policy = {"version": 1, "frontier": "big", "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
              "stages": [{"models": ["a"], "check": "judge"}, {"models": ["b", "c"], "check": "agreement"},
                         {"models": ["d"], "check": "none"}]}
    made = []

    async def call(model):
        made.append(model)
        if model == "c":
            await asyncio.sleep(30)
        return {"answer": "x", "cost_usd": 1.0}
    outcome = asyncio.run(run_stages_async(policy, call, checks={"judge": lambda result: {"passed": False, "cost_usd": 0.5}},
                                           deadline_seconds=0.1))
    assert made == ["a", "b", "c"] and not outcome["accepted"]  # the last stage is never started
    assert outcome["calls"] == [{"model": "a", "cost_usd": 1.0, "error": None},
                                {"model": "b", "cost_usd": 1.0, "error": None},
                                {"model": "c", "cost_usd": None, "error": "DeadlineExceeded"}]
    assert outcome["checks"][0]["cost_usd"] == 0.5

    async def slow_check(result):
        await asyncio.sleep(30)
    outcome = asyncio.run(run_stages_async(policy, call, checks={"judge": slow_check}, deadline_seconds=0.1))
    assert outcome["checks"] == [{"check": "judge", "model": "a", "passed": False, "cost_usd": None,
                                  "error": "DeadlineExceeded"}]


def test_a_check_that_cannot_pass_skips_its_stage_before_any_model_is_paid():
    policy = {"version": 1, "frontier": "big", "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
              "stages": [{"models": ["a"], "check": "judge"}, {"models": ["a"], "check": "none"}]}
    made = []

    async def call(model):
        made.append(model)
        return {"answer": "x", "cost_usd": 1.0}

    def judge(result):
        return {"passed": True, "cost_usd": 0.5}
    judge.skip_reason = "request_too_long"
    outcome = asyncio.run(run_stages_async(policy, call, checks={"judge": judge}))
    # The skipped stage called nothing, so the later stage still asks its model once.
    assert made == ["a"] and outcome["accepted"] and outcome["stage"] == 1
    assert outcome["checks"] == [{"check": "judge", "model": "a", "passed": False, "cost_usd": 0.0, "error": None,
                                  "reason": "request_too_long"}]
    judge.skip_reason = None
    made.clear()
    outcome = asyncio.run(run_stages_async(policy, call, checks={"judge": judge}))
    assert made == ["a"] and outcome["stage"] == 0 and "reason" not in outcome["checks"][0]


def test_freeze_policy_measured_shortlist_skips_a_cheap_pair_that_agrees_on_wrong_answers():
    """Two candidates: ``c+d`` is cheaper and its check passes everything, ``a+b``
    agrees less often but is always right when it does. With one candidate slot the
    default rule keeps ``c+d``, which cannot certify; the measured rule keeps ``a+b``."""
    frame, checks = recorded()
    rng = np.random.default_rng(1)
    cheap = frame[frame.model == "frontier"].assign(model="c+d", cost_usd=0.5)
    cheap["success"] = rng.random(len(cheap)) < 0.85
    frame = pd.concat([frame, cheap], ignore_index=True)
    checks = pd.concat([checks, cheap[["task_id", "model"]].assign(passed=True)], ignore_index=True)
    default = freeze_policy(frame, "frontier", checks, epsilon=0.02, top_k=1)
    measured = freeze_policy(frame, "frontier", checks, epsilon=0.02, top_k=1, shortlist="measured")
    assert default["stages"] == [] and not default["guarantee"]["certified"]
    assert measured["stages"] == [{"models": ["a", "b"], "check": "agreement"}]
    assert measured["guarantee"]["certified"] and measured["guarantee"]["bound"] <= 0.02
    # With room for both candidates the default rule finds the pair too.
    assert freeze_policy(frame, "frontier", checks, epsilon=0.02)["stages"] == measured["stages"]


ROOT = Path(__file__).parents[1]
SHIPPED = [ROOT / "results" / "2026-11-14-xroutebench" / "policy-multiple-choice-triples-eps02.json",
           ROOT / "results" / "2026-11-14-api-judge" / "policy-code-sonnet-judge-eps04.json"]


def test_a_frozen_policy_carries_a_digest_that_shows_later_edits():
    frame, checks = recorded()
    written = json.loads(dumps(freeze_policy(frame, "frontier", checks, epsilon=0.02)))
    assert written["provenance"][DIGEST_FIELD] == policy_digest(written)
    assert freeze_state(written) == FREEZE_UNCHANGED
    # Key order and layout are not content: the digest is over canonical JSON.
    assert freeze_state(json.loads(json.dumps(dict(reversed(list(written.items())))))) == FREEZE_UNCHANGED
    for edit in ({"tenant": "someone-else"}, {"guarantee": {**written["guarantee"], "epsilon": 0.04}},
                 {"stages": []}):
        edited = {**written, **edit}
        assert validate_policy(edited) is edited  # it still loads, but is marked
        assert freeze_state(edited) == FREEZE_EDITED
    assert freeze_state({**written, "provenance": {**written["provenance"], DIGEST_FIELD: "0" * 64}}) == FREEZE_EDITED
    unsigned = {**written, "provenance": {k: v for k, v in written["provenance"].items() if k != DIGEST_FIELD}}
    assert freeze_state(unsigned) == FREEZE_NO_DIGEST and freeze_state(POLICY) == FREEZE_NO_DIGEST


def test_policies_shipped_before_digests_still_load():
    for path in SHIPPED:
        policy = validate_policy(json.loads(path.read_text(encoding="utf-8")))
        assert freeze_state(policy) == FREEZE_NO_DIGEST and "answer_pattern" not in policy


def test_freeze_policy_can_name_the_tenant_and_the_answer_reader():
    frame, checks = recorded()
    plain = freeze_policy(frame, "frontier", checks, epsilon=0.02)
    assert plain["tenant"] == "acme" and "answer_pattern" not in plain and "outcomes_tenant" not in plain["provenance"]
    named = freeze_policy(frame, "frontier", checks, epsilon=0.02, tenant="helpwise", answer_pattern=r"\(([A-D])\)")
    assert named["tenant"] == "helpwise" and named["provenance"]["outcomes_tenant"] == "acme"
    assert named["answer_pattern"] == r"\(([A-D])\)" and freeze_state(json.loads(dumps(named))) == FREEZE_UNCHANGED
    # Only the new fields and the digest differ: the choice and the evidence are the same.
    assert {key: named[key] for key in ("stages", "guarantee", "estimate")} == {
        key: plain[key] for key in ("stages", "guarantee", "estimate")}
    for pattern in ("(", "", 3):
        with pytest.raises(ValueError, match="answer_pattern"):
            freeze_policy(frame, "frontier", checks, epsilon=0.02, answer_pattern=pattern)
    with pytest.raises(ValueError, match="tenant"):
        freeze_policy(frame, "frontier", checks, epsilon=0.02, tenant=" ")


def test_freeze_policy_command_writes_the_tenant_and_reader_and_refuses_a_bad_pattern(tmp_path, capsys):
    frame, checks = recorded(tasks=300)
    write_outcomes(frame, tmp_path / "outcomes.csv")
    checks.to_csv(tmp_path / "checks.csv", index=False)
    common = ["freeze-policy", "--input", str(tmp_path / "outcomes.csv"), "--frontier", "frontier",
              "--checks", str(tmp_path / "checks.csv"), "--epsilon", "0.04"]
    main([*common, "--output", str(tmp_path / "policy.json"), "--tenant", "helpwise",
          "--answer-pattern", r"Answer: ([A-D])"])
    written = json.loads((tmp_path / "policy.json").read_text(encoding="utf-8"))
    assert written["tenant"] == "helpwise" and written["answer_pattern"] == "Answer: ([A-D])"
    assert freeze_state(written) == FREEZE_UNCHANGED
    with pytest.raises(SystemExit):
        main([*common, "--output", str(tmp_path / "bad.json"), "--answer-pattern", "Answer: ([A-D]"])
    assert "--answer-pattern" in capsys.readouterr().err and not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("guarantee", [
    {"epsilon": "0.02"}, {"epsilon": 0}, {"epsilon": 1}, {"epsilon": 1.5}, {"epsilon": float("nan")},
    {"delta": 0}, {"delta": True}, {"delta": "0.05"}, {"bound": "0.01"}, {"bound": 1.5}, {"bound": -0.1},
    {"bound": float("inf")}, {"bound": None}, {"epsilon": None}, {"delta": None}])
def test_a_certified_guarantee_must_state_numbers_in_range(guarantee):
    """A bound written as a string once skipped the bound check and still read as certified."""
    with pytest.raises(ValueError):
        validate_policy({**POLICY, "guarantee": {**POLICY["guarantee"], **guarantee}})


def test_a_policy_without_stages_need_not_state_a_bound_but_what_it_states_is_checked():
    empty = {"version": 1, "frontier": "big", "stages": []}
    for guarantee in ({"certified": False}, {"certified": False, "epsilon": 0.02, "delta": 0.05, "bound": None}):
        validate_policy({**empty, "guarantee": guarantee})
    with pytest.raises(ValueError, match="bound"):
        validate_policy({**empty, "guarantee": {"certified": True, "epsilon": 0.02, "delta": 0.05}})
    with pytest.raises(ValueError, match="epsilon"):
        validate_policy({**empty, "guarantee": {"certified": False, "epsilon": "0.5"}})
    with pytest.raises(ValueError, match="answer_pattern"):
        validate_policy({**POLICY, "answer_pattern": "("})
