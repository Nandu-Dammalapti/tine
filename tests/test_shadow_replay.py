import asyncio
import csv
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "tine_shadow_replay", Path(__file__).parents[1] / "scripts" / "shadow_replay.py")
replay_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_module)

POLICY = {"version": 1, "tenant": "acme", "frontier": "big", "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
          "stages": [{"models": ["a", "b"], "check": "agreement"}]}
PRICES = {"big": {"input_per_million": 10.0, "output_per_million": 10.0},
          "a": {"input_per_million": 1.0, "output_per_million": 1.0},
          "b": {"input_per_million": 1.0, "output_per_million": 1.0}}


@pytest.fixture
def evidence():
    rows = [{"tenant": "acme", "task_id": "t", "prompt": "question", "task_type": "quiz",
             "model": model, "success": True, "cost_usd": 0.0,
             "input_tokens": 100, "output_tokens": 10} for model in PRICES]
    return pd.DataFrame(rows), {("t", model): "A" for model in PRICES}


@pytest.mark.parametrize("mode", ["shadow", "route"])
@pytest.mark.parametrize("problem, message", [
    ("pending", "Pending success"),
    ("tenants", "one tenant"),
    ("policy_tenant", "policy tenant"),
    ("missing_outcome", "Missing recorded outcome"),
    ("missing_answer", "Missing recorded answer"),
    ("unknown_tokens", "known input_tokens and output_tokens"),
    ("empty", "nonempty unique"),
    ("duplicate", "nonempty unique"),
    ("unknown_task", "Missing recorded outcome"),
])
def test_replay_rejects_incomplete_or_ambiguous_evidence_before_starting(evidence, monkeypatch, mode, problem, message):
    frame, answers = evidence
    policy, tasks = dict(POLICY), ["t"]
    if problem == "pending":
        frame["success"] = frame.success.astype(object)
        frame.loc[0, "success"] = None
    elif problem == "tenants":
        frame = pd.concat([frame, frame.assign(tenant="other")], ignore_index=True)
    elif problem == "policy_tenant":
        policy["tenant"] = "other"
    elif problem == "missing_outcome":
        frame = frame.loc[frame.model != "a"]
    elif problem == "missing_answer":
        del answers[("t", "big")]
    elif problem == "unknown_tokens":
        frame.loc[0, "input_tokens"] = None
    elif problem == "empty":
        tasks = []
    elif problem == "duplicate":
        tasks = ["t", "t"]
    elif problem == "unknown_task":
        tasks = ["unknown"]

    def unexpected_app(*args, **kwargs):
        raise AssertionError("Invalid evidence must be rejected before proxy startup")
    monkeypatch.setattr(replay_module, "create_app", unexpected_app)
    with pytest.raises(ValueError, match=message):
        asyncio.run(replay_module.replay(frame, answers, policy, PRICES, tasks, mode=mode))


def test_zero_priced_replay_has_no_defined_savings_fraction(evidence):
    frame, answers = evidence
    prices = {model: {"input_per_million": 0.0, "output_per_million": 0.0} for model in PRICES}
    result = asyncio.run(replay_module.replay(frame, answers, POLICY, prices, ["t"], mode="route"))
    assert result["ledger"]["cost_usd"] == result["ledger"]["frontier_cost_usd"] == 0.0
    assert result["ledger"]["savings_fraction"] is None
    assert result["graded"]["policy_accuracy"] == 1.0


@pytest.mark.parametrize("mode", ["shadow", "route"])
@pytest.mark.parametrize("recorded_tenant", ["acme", "deployment", "unrelated"])
def test_replay_honors_the_frozen_source_and_deployment_tenants(evidence, mode, recorded_tenant):
    frame, answers = evidence
    frame["tenant"] = recorded_tenant
    policy = {**POLICY, "tenant": "deployment", "provenance": {"outcomes_tenant": "acme"}}
    run = replay_module.replay(frame, answers, policy, PRICES, ["t"], mode=mode)
    if recorded_tenant == "unrelated":
        with pytest.raises(ValueError, match="policy tenant"):
            asyncio.run(run)
    else:
        result = asyncio.run(run)
        assert result["requests"] == 1
        assert result["graded"] == {"policy_accuracy": 1.0, "frontier_accuracy": 1.0, "degradation_rate": 0.0}


@pytest.mark.parametrize("contents, message", [
    ("task_id,model,answer\nt,big,A\nt,big,B\n", "Duplicate recorded answer"),
    ("task_id,model,answer\nt,big,A,extra\n", "fields; expected"),
    ("task_id,model,answer\nt,big\n", "fields; expected"),
    ("task_id,model,answer,answer\nt,big,A,B\n", "unique columns"),
    ("task_id,model\nt,big\n", "require task_id"),
])
def test_recorded_answer_csv_rejects_ambiguous_evidence(tmp_path, contents, message):
    path = tmp_path / "answers.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        replay_module.read_answers(path)


def test_recorded_answer_csv_preserves_literal_text_and_empty_answers(tmp_path):
    path = tmp_path / "answers.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerows([("task_id", "model", "answer"), ("001", "NA", ""),
                          ("null", "big", 'Explain "λ",\nthen answer')])
    assert replay_module.read_answers(path) == {("001", "NA"): "", ("null", "big"): 'Explain "λ",\nthen answer'}


def test_replay_reports_savings_and_grades_from_recorded_answers():
    truth = "ABCD"
    given = {"big": "ABCA", "a": "ABDD", "b": "ABCD"}  # the pair agrees on tasks 1, 2 and 4
    rows, answers = [], {}
    for model, letters in given.items():
        for index, letter in enumerate(letters):
            task_id = f"t{index}"
            rows.append({"tenant": "acme", "task_id": task_id, "prompt": f"question {index}", "task_type": "quiz",
                         "model": model, "success": letter == truth[index], "cost_usd": 0.0,
                         "input_tokens": 1_000_000, "output_tokens": 0})
            answers[(task_id, model)] = letter
    frame = pd.DataFrame(rows)
    tasks = [f"t{index}" for index in range(4)]
    routed = asyncio.run(replay_module.replay(frame, answers, POLICY, PRICES, tasks, mode="route"))
    assert routed["ledger"]["served_by_cheaper_models_share"] == 0.75
    assert routed["ledger"]["cost_usd"] == pytest.approx(18.0) and routed["ledger"]["savings_fraction"] == pytest.approx(0.55)
    assert routed["graded"] == {"policy_accuracy": 1.0, "frontier_accuracy": 0.75, "degradation_rate": 0.0}
    result = asyncio.run(replay_module.replay(pd.DataFrame(rows), answers, POLICY, PRICES,
                                              [f"t{index}" for index in range(4)]))
    report = result["shadow_report"]
    assert report["shadowed_requests"] == 4 and report["policy_would_accept"] == 3
    assert report["accepted_matching_frontier"] == 2  # task 4: the pair says D, the frontier said A
    # Frontier: 4 x $10. Policy: 4 x $2 for the pair, plus $10 on the one task it escalates.
    assert report["projected_policy_cost_usd"] == pytest.approx(18.0)
    assert report["projected_savings_fraction"] == pytest.approx(1 - 18 / 40)
    # Task 3 escalates to the frontier (right); on task 4 the pair is right and the frontier wrong.
    assert result["graded"] == {"policy_accuracy": 1.0, "frontier_accuracy": 0.75, "degradation_rate": 0.0}
