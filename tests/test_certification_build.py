import csv
import json

import pytest

from scripts.certification_build import main
from tine.policy import FREEZE_UNCHANGED, freeze_state, validate_policy


@pytest.fixture
def agreement_args(tmp_path):
    outcomes = tmp_path / "outcomes.jsonl"
    answers = tmp_path / "answers.csv"
    models = [
        ("cheap-a", 0.04, r"\dfrac{1}{2}"),
        ("cheap-b", 0.06, r"\frac{1}{2}"),
        ("frontier", 1.0, r"\frac{1}{2}"),
    ]
    with outcomes.open("w", encoding="utf-8") as recorded, answers.open("w", newline="", encoding="utf-8") as results:
        writer = csv.writer(results)
        writer.writerow(["task_id", "model", "answer"])
        for index in range(400):
            for model, cost, answer in models:
                row = {"tenant": "demo", "task_id": f"task-{index}", "prompt": f"Private question {index}",
                       "task_type": "math", "model": model, "success": True, "cost_usd": cost}
                recorded.write(json.dumps(row) + "\n")
                writer.writerow([row["task_id"], model, answer])
    return ["--input", str(outcomes), "--answers", str(answers), "--frontier", "frontier",
            "--answer-compare", "math"]


def test_build_freezes_both_limits_and_reports_frontier_fallback(tmp_path, agreement_args):
    output_dir = tmp_path / "build"
    pattern = r"Answer:\s*(.+)"
    main([*agreement_args, "--answer-pattern", pattern, "--output-dir", str(output_dir)])

    summary = json.loads((output_dir / "summary.json").read_text())
    assert [policy["guarantee"]["epsilon"] for policy in summary["policies"]] == [0.04, 0.02]
    for filename, result in zip(("policy-eps04.json", "policy-eps02.json"), summary["policies"], strict=True):
        policy = validate_policy(json.loads((output_dir / filename).read_text()))
        assert result["file"] == filename
        assert freeze_state(policy) == FREEZE_UNCHANGED
        assert policy["answer_compare"] == "math"
        assert policy["answer_pattern"] == pattern
        for key in ("guarantee", "estimate", "stages"):
            assert result[key] == policy[key]

    certified, fallback = summary["policies"]
    assert certified["guarantee"]["certified"]
    assert certified["stages"] == [{"models": ["cheap-a", "cheap-b"], "check": "agreement"}]
    assert certified["estimate"]["savings"] == pytest.approx(0.9)
    assert not fallback["guarantee"]["certified"]
    assert fallback["stages"] == []
    assert fallback["estimate"]["savings"] == 0
    assert "Private question" not in (output_dir / "summary.json").read_text()


def test_build_missing_answer_reader_leaves_existing_outputs_intact(tmp_path, agreement_args, capsys):
    output_dir = tmp_path / "build"
    output_dir.mkdir()
    old = {"policy-eps04.json": b"previous 4% policy", "policy-eps02.json": b"previous 2% policy",
           "summary.json": b"previous summary"}
    for name, contents in old.items():
        (output_dir / name).write_bytes(contents)

    with pytest.raises((SystemExit, ValueError)) as error:
        main([*agreement_args, "--output-dir", str(output_dir)])

    assert "answer-pattern" in (str(error.value) + capsys.readouterr().err)
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == old


def test_build_requires_complete_results_for_a_selected_named_check(tmp_path, capsys):
    outcomes = tmp_path / "outcomes.jsonl"
    checks = tmp_path / "checks.csv"
    output_dir = tmp_path / "build"
    check_rows = ["task_id,model,passed"]
    with outcomes.open("w", encoding="utf-8") as recorded:
        for index in range(400):
            base = {"tenant": "demo", "task_id": f"task-{index}", "prompt": f"Question {index}",
                    "task_type": "prepared"}
            passed = index != 0
            rows = [
                {**base, "model": "frontier", "success": True, "cost_usd": 1.0},
                {**base, "model": "cheap", "success": passed, "cost_usd": 0.1},
            ]
            recorded.writelines(json.dumps(row) + "\n" for row in rows)
            check_rows.append(f"task-{index},cheap,{passed}")
    checks.write_text("\n".join(check_rows) + "\n")
    argv = ["--input", str(outcomes), "--checks", str(checks), "--frontier", "frontier",
            "--single-check", "recorded-verifier", "--output-dir", str(output_dir)]

    main(argv)
    policy = json.loads((output_dir / "policy-eps04.json").read_text())
    assert policy["stages"] == [{"models": ["cheap"], "check": "recorded-verifier"}]
    saved = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    checks.write_text("\n".join(check_rows[:-1]) + "\n")
    with pytest.raises(SystemExit):
        main(argv)

    assert "Selected check for cheap is missing results" in capsys.readouterr().err
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == saved
