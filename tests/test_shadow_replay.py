import asyncio
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "tine_shadow_replay", Path(__file__).parents[1] / "scripts" / "shadow_replay.py")
replay_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_module)

POLICY = {"version": 1, "tenant": "acme", "frontier": "big", "guarantee": {"certified": True},
          "stages": [{"models": ["a", "b"], "check": "agreement"}]}
PRICES = {"big": {"input_per_million": 10.0, "output_per_million": 10.0},
          "a": {"input_per_million": 1.0, "output_per_million": 1.0},
          "b": {"input_per_million": 1.0, "output_per_million": 1.0}}


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
