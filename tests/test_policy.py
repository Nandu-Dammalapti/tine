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
    DIGEST_FIELD, FREEZE_EDITED, FREEZE_NO_DIGEST, FREEZE_UNCHANGED, answer_mode, canonical_answer, dumps,
    freeze_policy, freeze_state, policy_digest, run_policy, run_stages_async, same_answer, validate_policy,
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


def test_a_chain_certified_with_a_check_cannot_be_frozen_as_unchecked():
    """The recorded check rejected some of the cheap model's answers, so the chain was
    certified with it. Left at the default, the policy would serve them all unchecked."""
    frame, checks = recorded()
    frame["model"] = frame.model.replace({"a+b": "cheap"})
    checks["model"] = "cheap"
    with pytest.raises(ValueError, match="single_check is 'none'"):
        freeze_policy(frame, "frontier", checks, epsilon=0.02)
    # A model whose recorded check passed every answer really is served as it is.
    frame.loc[frame.model == "cheap", "success"] = True
    checks["passed"] = True
    policy = freeze_policy(frame, "frontier", checks, epsilon=0.02)
    assert policy["stages"] == [{"models": ["cheap"], "check": "none"}]


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
                                  "error": "DeadlineExceeded", "checked_model": "a"}]


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
    # No model answered the skipped stage, so no answer was checked.
    assert outcome["checks"] == [{"check": "judge", "model": "a", "passed": False, "cost_usd": 0.0, "error": None,
                                  "reason": "request_too_long", "checked_model": None}]
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


@pytest.mark.parametrize("same", [
    [r"14.5\%", "14.5%", "14.50", "14.5", "$14.5$", r"14.5\%."],
    [r"\dfrac{1}{2}", r"\frac{1}{2}", r"\tfrac{1}{2}", r"\frac { 1 } { 2 }"],
    ["1,234", "1234", r"\$1,234", "1234.0", r"1\,234"],
    [r"90^\circ", r"90^{\circ}", "90"],
    [r"\text{(B)}", "(B)", r"\text{ (B) }"],
    [r"\left( 1, 2 \right)", "(1,2)", r"(1,\!2)", r"(1,\ 2)"],
    ["0", "-0.0", "0.00", "+0"],
    [".5", "0.5", "0.50"],
])
def test_math_comparison_reads_one_value_written_several_ways(same):
    forms = {canonical_answer(answer, "math") for answer in same}
    assert len(forms) == 1 and None not in forms and same_answer(same, "math")
    # Exact comparison, the default, keeps today's behaviour: these differ as text.
    assert not same_answer(same)


@pytest.mark.parametrize("different", [
    ["14.5", "14.51"], ["0.5", r"\frac{1}{2}"], [r"9.9\%", "0.099"], ["1e3", "1000"], [r"\frac12", r"\frac{1}{2}"],
    ["12345678901234567", "12345678901234568"], ["1,2345", "12345"], ["x+1", "1+x"], ["5", r"5\text{ cm}"],
    [r"\rightarrow", "ow"],
])
def test_math_comparison_never_joins_different_values_or_notations_it_does_not_know(different):
    """Agreement must mean the same value: no tolerance, and nothing it cannot read."""
    assert not same_answer(different, "math")


def test_canonical_answer_keeps_missing_answers_missing_and_exact_mode_as_it_was():
    for missing in (None, "", "  ", "None", "[None]", "null", 3, "$ $", "$$", "."):
        assert canonical_answer(missing, "math") is None
    assert canonical_answer("  a \n b ") == "a b" and canonical_answer(r"\dfrac{1}{2}") == r"\dfrac{1}{2}"
    assert canonical_answer("14.50", "math") == "14.5" and canonical_answer("100", "math") == "100"
    assert canonical_answer(r"\$1,234.50", "math") == "1234.5" and canonical_answer(".5", "math") == "0.5"
    assert not same_answer([None, None], "math") and not same_answer(["$", "$"], "math")
    with pytest.raises(ValueError, match="answer comparison"):
        canonical_answer("1", "numeric")


def test_a_policy_names_how_its_agreement_compares_answers():
    assert answer_mode(POLICY) == "exact" and answer_mode({**POLICY, "answer_compare": "math"}) == "math"
    for mode in ("exact", "math", None):
        validate_policy({**POLICY, "answer_compare": mode})
    for bad in ("Math", "", "numeric", 1, ["math"]):
        with pytest.raises(ValueError, match="answer_compare"):
            validate_policy({**POLICY, "answer_compare": bad})


def test_run_policy_agrees_by_value_only_when_the_policy_says_so():
    answers = {"a": r"14.5\%", "b": "14.50", "c": "never", "big": "14.5"}
    call, made = caller(answers)
    assert run_policy(POLICY, call)["served_by"] == "c" and made == ["a", "b", "c"]
    call, made = caller(answers)
    outcome = run_policy({**POLICY, "answer_compare": "math"}, call)
    assert outcome["served_by"] == "a" and outcome["result"]["answer"] == r"14.5\%" and made == ["a", "b"]
    # A caller's own agreement test still wins.
    call, _ = caller(answers)
    assert run_policy({**POLICY, "answer_compare": "math"}, call, agree=lambda found: False)["served_by"] == "c"

    async def call_async(model):
        return {"answer": answers[model], "cost_usd": 1.0}
    routed = asyncio.run(run_stages_async({**POLICY, "answer_compare": "math"}, call_async))
    assert routed["accepted"] and routed["stage"] == 0
    assert asyncio.run(run_stages_async(POLICY, call_async))["stage"] == 1


def test_freeze_policy_writes_the_answer_comparison_only_when_asked(tmp_path, capsys):
    frame, checks = recorded()
    plain = freeze_policy(frame, "frontier", checks, epsilon=0.02)
    assert "answer_compare" not in plain and answer_mode(plain) == "exact"
    math_policy = freeze_policy(frame, "frontier", checks, epsilon=0.02, answer_compare="math")
    assert math_policy["answer_compare"] == "math" and freeze_state(json.loads(dumps(math_policy))) == FREEZE_UNCHANGED
    assert {key: math_policy[key] for key in ("stages", "guarantee", "estimate")} == {
        key: plain[key] for key in ("stages", "guarantee", "estimate")}
    with pytest.raises(ValueError, match="answer_compare"):
        freeze_policy(frame, "frontier", checks, epsilon=0.02, answer_compare="numeric")
    small, small_checks = recorded(tasks=300)
    write_outcomes(small, tmp_path / "outcomes.csv")
    small_checks.to_csv(tmp_path / "checks.csv", index=False)
    common = ["freeze-policy", "--input", str(tmp_path / "outcomes.csv"), "--frontier", "frontier",
              "--checks", str(tmp_path / "checks.csv"), "--epsilon", "0.04"]
    main([*common, "--output", str(tmp_path / "math.json"), "--answer-compare", "math"])
    main([*common, "--output", str(tmp_path / "plain.json")])
    assert json.loads((tmp_path / "math.json").read_text(encoding="utf-8"))["answer_compare"] == "math"
    assert "answer_compare" not in json.loads((tmp_path / "plain.json").read_text(encoding="utf-8"))
    with pytest.raises(SystemExit):
        main([*common, "--output", str(tmp_path / "bad.json"), "--answer-compare", "numeric"])
    assert "--answer-compare" in capsys.readouterr().err and not (tmp_path / "bad.json").exists()


def recorded_with_matches(tasks=600, seed=0):
    """``recorded``, with a pair that is sometimes wrong when it agrees and checks that
    also say whether the pair's answer equals the frontier's. When both are right the
    answers are equal one time in twenty less than always (the same value written two
    ways); when exactly one is right they differ; when both are wrong they differ half
    the time."""
    rng = np.random.default_rng(seed)
    rows, checks = [], []
    for index in range(tasks):
        task_id = f"t{index}"
        agree, right, frontier_right = rng.random() < 0.7, rng.random() < 0.99, rng.random() < 0.9
        if right and frontier_right:
            matches = rng.random() < 0.95
        else:
            matches = not right and not frontier_right and rng.random() < 0.5
        base = {"tenant": "acme", "task_id": task_id, "prompt": f"question {index}", "task_type": "quiz"}
        rows.append({**base, "model": "frontier", "success": bool(frontier_right), "cost_usd": 10.0})
        rows.append({**base, "model": "a+b", "success": bool(right), "cost_usd": 1.0})
        checks.append({"task_id": task_id, "model": "a+b", "passed": bool(agree), "matches_frontier": bool(matches)})
    return pd.DataFrame(rows), pd.DataFrame(checks)


def test_freeze_policy_records_how_often_accepted_answers_differed_from_the_frontier_without_a_loss():
    frame, checks = recorded_with_matches()
    policy = freeze_policy(frame, "frontier", checks, epsilon=0.04)
    assert policy["stages"] == [{"models": ["a", "b"], "check": "agreement"}]
    block = policy["drift_baseline"]
    assert list(policy)[-2:] == ["drift_baseline", "provenance"]
    assert block["parts"] == ["certify", "test"] and block["answer_compare"] == "exact"
    assert set(block["by_part"]) == {"choose", "certify", "test"}
    # Counted here from the same rows, without the split: the three parts hold every task once.
    table = checks.merge(frame.pivot(index="task_id", columns="model", values="success"), on="task_id")
    accepted = table[table.passed]
    differing = accepted[~accepted.matches_frontier]
    lost = differing[differing.frontier & ~differing["a+b"]]
    parts = block["by_part"].values()
    assert sum(part["tasks"] for part in parts) == len(table) == 600
    assert sum(part["compared"] for part in parts) == len(accepted)
    assert sum(part["differing"] for part in parts) == len(differing) > len(lost) > 0
    assert sum(part["differing_not_lost"] for part in parts) == len(differing) - len(lost)
    # The drift check uses the certify and test parts together; the choose part picked the chain.
    for name in ("tasks", "compared", "differing", "differing_not_lost"):
        assert block[name] == block["by_part"]["certify"][name] + block["by_part"]["test"][name]
    assert block["by_part"]["certify"]["tasks"] == policy["guarantee"]["certify_tasks"]
    assert block["by_part"]["test"]["tasks"] == policy["estimate"]["test_tasks"]
    assert block["accepted"] == block["compared"] and block["tasks"] > block["compared"]
    assert block["differing"] == block["differing_lost"] + block["differing_not_lost"]
    # A loss is a differing answer, so the test part's loss is counted among them.
    test_part = block["by_part"]["test"]
    assert (test_part["differing"] - test_part["differing_not_lost"]) / test_part["tasks"] == pytest.approx(
        policy["estimate"]["degradation_rate"])
    # The block is covered by the freeze digest, and a policy that carries it loads.
    written = json.loads(dumps(policy))
    assert freeze_state(written) == FREEZE_UNCHANGED and validate_policy(written) is written
    edited = {**written, "drift_baseline": {**block, "differing_not_lost": block["differing"]}}
    assert validate_policy(edited) is edited and freeze_state(edited) == FREEZE_EDITED
    # The comparison named in the block is the policy's own.
    by_value = freeze_policy(frame, "frontier", checks, epsilon=0.04, answer_compare="math")
    assert by_value["drift_baseline"] == {**block, "answer_compare": "math"}


def test_checks_that_do_not_say_whether_answers_match_freeze_the_policy_as_before():
    frame, checks = recorded_with_matches()
    with_block = freeze_policy(frame, "frontier", checks, epsilon=0.04)
    without = freeze_policy(frame, "frontier", checks.drop(columns="matches_frontier"), epsilon=0.04)
    assert "drift_baseline" not in without and list(without) == [key for key in with_block if key != "drift_baseline"]
    for key in without:
        if key != "provenance":
            assert without[key] == with_block[key], key
    assert {key: value for key, value in without["provenance"].items() if key not in ("created", DIGEST_FIELD)} == {
        key: value for key, value in with_block["provenance"].items() if key not in ("created", DIGEST_FIELD)}
    # One accepted answer whose match is not known: no baseline is better than a wrong one.
    unknown = checks.astype({"matches_frontier": object})
    unknown.loc[unknown.index[unknown.passed][0], "matches_frontier"] = None
    assert "drift_baseline" not in freeze_policy(frame, "frontier", unknown, epsilon=0.04)
    # An unknown match on a task the pair did not accept does not matter.
    unknown = checks.astype({"matches_frontier": object})
    unknown.loc[~unknown.passed, "matches_frontier"] = None
    assert freeze_policy(frame, "frontier", unknown, epsilon=0.04)["drift_baseline"] == with_block["drift_baseline"]
    # Nothing certified: no stage, so nothing to compare live.
    frame.loc[frame.model == "a+b", "success"] = False
    assert "drift_baseline" not in freeze_policy(frame, "frontier", checks, epsilon=0.04)


def test_the_drift_baseline_walks_the_chain_and_does_not_compare_answers_a_named_check_accepted():
    from tine.policy import _drift_baseline

    # Columns: the pair a+b, the single model c, the frontier. Eight tasks.
    outcomes = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]], bool)
    passes = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]], bool)
    matches = np.array([[True, None, None], [False, None, None], [False, None, None], [False, None, None],
                        [None, False, None], [None, True, None], [False, None, None], [None, False, None]], object)
    parts = {"choose": np.array([1, 1, 0, 0, 0, 0, 0, 0], bool), "certify": np.array([0, 0, 1, 1, 1, 1, 0, 0], bool),
             "test": np.array([0, 0, 0, 0, 0, 0, 1, 1], bool)}
    stages = [{"models": ["a", "b"], "check": "agreement"}, {"models": ["c"], "check": "none"}]
    block = _drift_baseline([0, 1], stages, outcomes, passes, matches, 2, parts, "exact")
    # Task 1 is a loss in the choose part. In the other two parts tasks 2 (the frontier is wrong), 3 (both wrong),
    # 4 and 6 (both right, written differently) differ without a loss, and task 7 is a loss.
    assert block == {
        "parts": ["certify", "test"], "answer_compare": "exact", "tasks": 6, "accepted": 6, "compared": 6,
        "differing": 5, "differing_lost": 1, "differing_not_lost": 4,
        "by_part": {"choose": {"tasks": 2, "compared": 2, "differing": 1, "differing_not_lost": 0},
                    "certify": {"tasks": 4, "compared": 4, "differing": 3, "differing_not_lost": 3},
                    "test": {"tasks": 2, "compared": 2, "differing": 2, "differing_not_lost": 1}}}
    # With a named check on the single model its accepted answers are not compared, live or here.
    # Their tasks still count: the quality limit is a share of all tasks.
    judged = [stages[0], {"models": ["c"], "check": "tests+judge"}]
    block = _drift_baseline([0, 1], judged, outcomes, passes, matches, 2, parts, "math")
    assert (block["tasks"], block["accepted"], block["compared"], block["differing"]) == (6, 6, 3, 3)
    assert block["differing_not_lost"] == 3 and block["by_part"]["test"] == {
        "tasks": 2, "compared": 1, "differing": 1, "differing_not_lost": 1}
    # No block: no stage, no column, no compared stage, or a compared accepted answer of unknown match.
    assert _drift_baseline([], [], outcomes, passes, matches, 2, parts, "exact") is None
    assert _drift_baseline([0, 1], stages, outcomes, passes, None, 2, parts, "exact") is None
    assert _drift_baseline([1], judged[1:], outcomes, passes, matches, 2, parts, "exact") is None
    matches[5, 1] = None
    assert _drift_baseline([0, 1], stages, outcomes, passes, matches, 2, parts, "exact") is None
    assert _drift_baseline([0, 1], judged, outcomes, passes, matches, 2, parts, "exact") is not None


@pytest.mark.parametrize("block", [
    None, [], "7 of 692", {}, {"tasks": 692}, {"tasks": 692, "differing": 16},
    {"tasks": 692, "differing": 16, "differing_not_lost": 7},  # no count of compared answers
    {"tasks": 0, "compared": 0, "differing": 0, "differing_not_lost": 0},
    {"tasks": 692, "compared": 612, "differing": 16, "differing_not_lost": 17},
    {"tasks": 10, "compared": 10, "differing": 11, "differing_not_lost": 7},
    {"tasks": 692, "compared": 15, "differing": 16, "differing_not_lost": 7},
    {"tasks": 692, "compared": 693, "differing": 16, "differing_not_lost": 7},
    {"tasks": 692, "compared": 612, "differing": 16, "differing_not_lost": -1},
    {"tasks": 692.0, "compared": 612, "differing": 16, "differing_not_lost": 7},
    {"tasks": True, "compared": 1, "differing": 1, "differing_not_lost": 1},
    {"tasks": "692", "compared": "612", "differing": "16", "differing_not_lost": "7"},
    # More tasks than any history holds: the bound worked out from such a count would fail.
    {"tasks": 10**30, "compared": 0, "differing": 0, "differing_not_lost": 0},
    {"tasks": 10**9 + 1, "compared": 5, "differing": 5, "differing_not_lost": 5}])
def test_a_malformed_drift_baseline_is_refused(block):
    with pytest.raises(ValueError, match="drift_baseline"):
        validate_policy({**POLICY, "drift_baseline": block})


def test_a_policy_loads_with_a_drift_baseline_and_without_one():
    assert validate_policy(POLICY) is POLICY and "drift_baseline" not in POLICY
    from tine.shadow import drift_allowance

    for block in ({"tasks": 692, "compared": 612, "differing": 16, "differing_not_lost": 7},
                  {"tasks": 1, "compared": 0, "differing": 0, "differing_not_lost": 0},
                  {"tasks": 3, "compared": 3, "differing": 3, "differing_not_lost": 3},
                  {"tasks": 10**9, "compared": 10**9, "differing": 10**9, "differing_not_lost": 0}):
        validate_policy({**POLICY, "drift_baseline": block})
        assert 0.04 < drift_allowance(0.04, block)["drift_allowance"] <= 1  # every count that loads gives a level


def test_freeze_policy_command_reads_the_match_column_and_blank_values(tmp_path):
    frame, checks = recorded_with_matches()
    write_outcomes(frame, tmp_path / "outcomes.csv")
    checks.to_csv(tmp_path / "checks.csv", index=False)
    blank = checks.astype({"matches_frontier": object})
    blank.loc[~blank.passed, "matches_frontier"] = None  # written as an empty field
    blank.to_csv(tmp_path / "blank.csv", index=False)
    checks.drop(columns="matches_frontier").to_csv(tmp_path / "plain.csv", index=False)
    written = {}
    for name in ("checks", "blank", "plain"):
        main(["freeze-policy", "--input", str(tmp_path / "outcomes.csv"), "--frontier", "frontier",
              "--checks", str(tmp_path / f"{name}.csv"), "--epsilon", "0.04", "--output", str(tmp_path / f"{name}.json")])
        written[name] = json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))
        assert freeze_state(written[name]) == FREEZE_UNCHANGED
    assert written["checks"]["stages"] and written["checks"]["drift_baseline"]["differing"] > 0
    assert written["blank"]["drift_baseline"] == written["checks"]["drift_baseline"]
    assert "drift_baseline" not in written["plain"]
    assert {key: written["plain"][key] for key in ("stages", "guarantee", "estimate")} == {
        key: written["checks"][key] for key in ("stages", "guarantee", "estimate")}
