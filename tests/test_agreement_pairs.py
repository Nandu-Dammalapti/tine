import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from tine.cascade import run_cascade_experiment
from tine.checks import read_checks
from tine.data import read_outcomes

spec = importlib.util.spec_from_file_location(
    "tine_agreement", Path(__file__).parents[1] / "scripts" / "agreement_pairs.py")
agreement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agreement)

# Per task: the right answer, then each model's answer. "big" is the frontier.
ANSWERS = {"cheap": "AABCA", "small": "AABDB", "tiny": "ABBD ", "big": "AABCC"}
TRUTH = "AABCC"
COSTS = {"cheap": 1.0, "small": 2.0, "tiny": 0.5, "big": 10.0}


def outcomes_frame():
    rows = []
    for model, answers in ANSWERS.items():
        for index, answer in enumerate(answers, 1):
            rows.append({"tenant": "t", "task_id": f"quiz:test:{index}", "prompt": f"Question {index}?",
                         "task_type": "quiz", "model": model, "success": answer == TRUTH[index - 1],
                         "cost_usd": COSTS[model], "input_tokens": 10, "output_tokens": 5})
    return pd.DataFrame(rows)


def recorded():
    return {(f"quiz:test:{index}", model): agreement.normalize(answer)
            for model, answers in ANSWERS.items() for index, answer in enumerate(answers, 1)}


def test_normalize_treats_blank_and_placeholder_answers_as_missing():
    assert agreement.normalize(" B \n") == "B" and agreement.normalize([[1, 2]]) == "[[1, 2]]"
    for missing in [None, "", "   ", "None", "[None]"]:
        assert agreement.normalize(missing) is None


def test_pairs_cost_both_calls_and_pass_only_when_answers_match():
    frame, checks, report = agreement.build_pairs(outcomes_frame(), recorded(), "big", max_cost_share=0.5)
    assert report["pairs"] == 3 and set(frame.model) == {"big", "cheap+small", "cheap+tiny", "small+tiny"}
    pair = frame[frame.model == "cheap+small"].set_index("task_id")
    assert (pair.cost_usd == 3.0).all() and (pair.input_tokens == 20).all()
    passed = {(row["task_id"], row["model"]): row["passed"] for row in checks}
    assert [passed[(f"quiz:test:{n}", "cheap+small")] for n in range(1, 6)] == [True, True, True, False, False]
    # tiny gave no answer on task 5: a missing answer never agrees.
    assert passed[("quiz:test:5", "small+tiny")] is False
    # Where the pair agrees its recorded success is the shared one.
    assert pair.success.tolist()[:3] == [True, True, True]
    assert report["agreement_rate"]["cheap+small"] == 0.6


def test_triples_require_every_member_and_respect_the_cost_cap():
    frame, checks, report = agreement.build_pairs(outcomes_frame(), recorded(), "big", group_size=3)
    assert report["pairs"] == 1 and "cheap+small+tiny" in set(frame.model)
    passed = [row["passed"] for row in checks]
    assert passed == [True, False, True, False, False]
    assert (frame[frame.model == "cheap+small+tiny"].cost_usd == 3.5).all()
    with pytest.raises(ValueError, match="cheap enough"):
        agreement.build_pairs(outcomes_frame(), recorded(), "big", max_cost_share=0.1)
    for bad in [{"frontier": "nobody"}, {"max_cost_share": 0}, {"group_size": 1}]:
        with pytest.raises(ValueError):
            agreement.build_pairs(outcomes_frame(), recorded(), **({"frontier": "big"} | bad))


def test_cli_output_feeds_the_cascade(tmp_path):
    rng_frame = []
    import numpy as np
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 4, 300)
    bench = tmp_path / "bench"
    for model, accuracy, cost in [("a", 0.8, 1.0), ("b", 0.8, 1.0), ("big", 0.95, 10.0)]:
        answers = np.where(rng.random(300) < accuracy, truth, (truth + 1 + rng.integers(0, 3, 300)) % 4)
        (bench / model).mkdir(parents=True)
        (bench / model / "run.json").write_text(json.dumps({
            "dataset_name": "quiz", "split": "test", "model_name": model,
            "records": [{"index": n + 1, "prediction": "ABCD"[answer]} for n, answer in enumerate(answers)]}))
        for n, answer in enumerate(answers):
            rng_frame.append({"tenant": "t", "task_id": f"quiz:test:{n + 1}", "prompt": f"Question {n}?",
                              "task_type": "quiz", "model": model, "success": bool(answer == truth[n]),
                              "cost_usd": cost, "input_tokens": 10, "output_tokens": 5})
    source = tmp_path / "outcomes.jsonl"
    pd.DataFrame(rng_frame).to_json(source, orient="records", lines=True)
    agreement.main(["--input", str(source), "--bench-dir", str(bench), "--frontier", "big",
                    "--output", str(tmp_path / "pairs.jsonl"), "--checks", str(tmp_path / "agree.csv")])
    result = run_cascade_experiment(read_outcomes(tmp_path / "pairs.jsonl"), "big", tmp_path / "run",
                                    checks=read_checks(tmp_path / "agree.csv"), seeds=range(2), epsilons=(0.05,))
    assert result["check_mode"] == "measured" and result["models"] == ["a+b", "big"]
    assert result["check_coverage"]["known_checks"] == 300
