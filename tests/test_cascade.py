import numpy as np
import pandas as pd
import pytest

from tine.__main__ import main
from tine.cascade import candidate_chains, chain_metrics, run_cascade_experiment, select_chain
from tine.data import synthetic_fixture

# Columns: 0 cheap, 1 mid, 2 frontier.
OUTCOMES = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
COSTS = np.tile([1.0, 2.0, 10.0], (4, 1))


def test_perfect_check_never_degrades_and_counts_every_attempt():
    m = chain_metrics([0, 1], OUTCOMES, COSTS, 2, catch=1.0)
    # task0: 1; task1: 1+2; task2: 1+2+10; task3: 1+2+10 -> 30 vs frontier 40
    assert m["savings"] == pytest.approx(0.25)
    assert m["degradation_rate"] == 0
    assert m["accuracy"] == 0.75 == m["frontier_accuracy"]
    assert m["frontier_call_rate"] == 0.5
    assert m["mean_calls"] == pytest.approx((1 + 2 + 3 + 3) / 4)


def test_missed_failures_ship_and_degrade():
    m = chain_metrics([0], OUTCOMES, COSTS, 2, catch=0.0)
    assert m["frontier_call_rate"] == 0
    assert m["accuracy"] == 0.25
    assert m["degradation_rate"] == 0.5  # tasks 1 and 2: frontier right, cheap wrong
    half = chain_metrics([0], OUTCOMES, COSTS, 2, catch=0.5)
    assert half["degradation_rate"] == pytest.approx(0.25)


def test_chain_validation():
    with pytest.raises(ValueError):
        chain_metrics([2], OUTCOMES, COSTS, 2)
    with pytest.raises(ValueError):
        chain_metrics([0, 0], OUTCOMES, COSTS, 2)
    with pytest.raises(ValueError):
        chain_metrics([0], OUTCOMES, COSTS, 2, catch=1.5)


def test_candidates_exclude_frontier_and_pricier_models():
    costs = np.tile([1.0, 20.0, 10.0], (4, 1))
    assert candidate_chains(OUTCOMES, costs, 2) == [[0]]
    assert [0, 1] in candidate_chains(OUTCOMES, COSTS, 2)


def test_select_respects_epsilon_and_falls_back():
    assert select_chain(OUTCOMES, COSTS, 2, catch=1.0, epsilon=0.0) != []
    assert select_chain(OUTCOMES, COSTS, 2, catch=0.0, epsilon=0.0) == []


def test_experiment_writes_held_out_summary(tmp_path):
    frame = synthetic_fixture(n_tasks=200, seed=5)
    result = run_cascade_experiment(frame, "frontier", tmp_path, seeds=range(3),
                                    epsilons=(0.02,), catches=(1.0, 0.9))
    summary = pd.read_csv(tmp_path / "cascade-summary.csv")
    assert len(summary) == 2 and result["complete_tasks"] == 200
    runs = pd.read_csv(tmp_path / "cascade-runs.csv")
    assert len(runs) == 6 and runs.chain.str.endswith("> frontier").all()
    assert (runs.loc[runs["catch"] == 1.0, "degradation_rate"] == 0).all()
    assert "hidden tests" in (tmp_path / "report.md").read_text()


def test_cli(tmp_path, capsys):
    from tine.data import write_outcomes
    write_outcomes(synthetic_fixture(n_tasks=120, seed=6), tmp_path / "o.jsonl")
    main(["cascade", "--input", str(tmp_path / "o.jsonl"), "--frontier", "frontier",
          "--output", str(tmp_path / "out"), "--seeds", "2", "--epsilon", "0.02", "--catch", "1.0"])
    assert "Report:" in capsys.readouterr().out
