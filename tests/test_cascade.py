import numpy as np
import pandas as pd
import pytest

from tine.__main__ import main
from tine.cascade import (candidate_chains, certify_chain, chain_metrics, run_cascade_experiment,
                          select_chain, simulated_passes)
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


@pytest.mark.parametrize("changes", [
    {"chain": [-1]}, {"chain": [True]}, {"frontier_index": True},
    {"outcomes": OUTCOMES.astype(float)}, {"costs": COSTS.astype(complex) + 2j},
    {"passes": np.full(OUTCOMES.shape, None)}, {"passes": OUTCOMES.astype(float)},
])
def test_cascade_rejects_ambiguous_measurements_and_indices(changes):
    arguments = {"chain": [0], "outcomes": OUTCOMES, "costs": COSTS, "frontier_index": 2}
    with pytest.raises(ValueError):
        chain_metrics(**(arguments | changes))


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
    assert len(summary) == 4 and result["complete_tasks"] == 200  # 2 catches x 2 methods
    assert set(summary.method) == {"uncertified", "certified"}
    runs = pd.read_csv(tmp_path / "cascade-runs.csv")
    assert len(runs) == 12 and runs.chain.str.endswith("frontier").all()
    assert (runs.loc[runs["catch"] == 1.0, "degradation_rate"] == 0).all()
    certified = runs[(runs.method == "certified") & (runs.chain != "frontier")]
    assert (certified.certified_bound <= 0.02).all()
    assert "hidden tests" in (tmp_path / "report.md").read_text()
    assert result["synthetic"] is True
    assert "SYNTHETIC outcomes" in (tmp_path / "report.md").read_text()


def test_failed_cascade_rerun_preserves_previous_report(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    previous = {"report.md": "Previous report", "cascade-runs.csv": "Previous runs",
                "cascade-summary.csv": "Previous summary", "notes.txt": "User notes"}
    for name, content in previous.items():
        (output / name).write_text(content)
    to_csv = pd.DataFrame.to_csv

    def fail_summary(frame, path, **kwargs):
        if path.name == "cascade-summary.csv":
            raise OSError("Disk write failed")
        return to_csv(frame, path, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_summary)
    with pytest.raises(OSError, match="Disk write failed"):
        run_cascade_experiment(synthetic_fixture(100), "frontier", output,
                               seeds=[42], catches=[1], epsilons=[0.02])
    assert {path.name: path.read_text() for path in output.iterdir()} == previous


def test_cli(tmp_path, capsys):
    from tine.data import write_outcomes
    write_outcomes(synthetic_fixture(n_tasks=120, seed=6), tmp_path / "o.jsonl")
    main(["cascade", "--input", str(tmp_path / "o.jsonl"), "--frontier", "frontier",
          "--output", str(tmp_path / "out"), "--seeds", "2", "--epsilon", "0.02", "--catch", "1.0"])
    assert "Report:" in capsys.readouterr().out


def test_observed_checks_drive_escalation():
    # cheap model's wrong answer on task 1 passes the check -> ships wrong.
    passes = OUTCOMES.copy()
    passes[1, 0] = True
    m = chain_metrics([0], OUTCOMES, COSTS, 2, passes=passes)
    assert m["degradation_rate"] == 0.25
    assert m["accuracy"] == 0.5  # task0 right; task1 shipped wrong; task2 escalated right; task3 none
    assert m["frontier_call_rate"] == 0.5


def test_simulated_passes_never_reject_right_answers():
    rng = np.random.default_rng(0)
    passes = simulated_passes(np.ones((50, 2), dtype=bool), 0.9, rng)
    assert passes.all()
    wrong = simulated_passes(np.zeros((5000, 1), dtype=bool), 0.9, rng)
    assert 0.07 < wrong.mean() < 0.13


def test_certification_falls_back_when_certify_part_disagrees():
    n = 400
    good = np.ones((n, 3), dtype=bool)
    costs = np.tile([1.0, 2.0, 10.0], (n, 1))
    bad = good.copy()
    bad[: n // 10, 0] = False  # cheap model wrong on 10% of the certify part
    passes_bad = bad.copy()
    passes_bad[:, 0] = True    # and the check misses it
    choose = (good, costs, good)
    chain, bound = certify_chain([[0]], choose, (bad, costs, passes_bad), 2, epsilon=0.02, delta=0.05)
    assert chain == [] and bound is None
    chain, bound = certify_chain([[0]], choose, choose, 2, epsilon=0.02, delta=0.05)
    assert chain == [0] and bound <= 0.02


def test_measured_checks_input(tmp_path):
    frame = synthetic_fixture(n_tasks=150, seed=7)
    checks = frame[["task_id", "model"]].assign(passed=frame.success.astype(bool))
    result = run_cascade_experiment(frame, "frontier", tmp_path, checks=checks, seeds=range(2),
                                    epsilons=(0.02,))
    assert {row["catch"] for row in result["summary"]} == {"measured"}
    assert result["check_mode"] == "measured"
    assert result["check_coverage"]["known_checks"] == len(frame)
    assert result["check_coverage"]["unknown_checks"] == 0
    assert "check execution costs are excluded" in result["cost_scope"]


@pytest.mark.parametrize("kwargs", [
    {"catches": [1.5]}, {"epsilons": [float("nan")]}, {"delta": 0},
    {"fractions": [0.5, 0.5]}, {"seeds": []}, {"epsilons": []},
    {"seeds": [True]}, {"max_length": 0},
])
def test_invalid_experiment_settings_fail_before_creating_output(tmp_path, kwargs):
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        run_cascade_experiment(synthetic_fixture(20), "frontier", output, **kwargs)
    assert not output.exists()


def test_zero_baseline_cost_has_unknown_savings_and_cannot_select_a_chain(tmp_path):
    import json

    costs = np.zeros_like(COSTS)
    assert select_chain(OUTCOMES, costs, 2, epsilon=0.02, chains=[[0]]) == []
    data = (OUTCOMES, costs, OUTCOMES)
    assert certify_chain([[0]], data, data, 2, epsilon=0.02, delta=0.05) == ([], None)
    frame = synthetic_fixture(100)
    frame["cost_usd"] = 0.0
    result = run_cascade_experiment(frame, "frontier", tmp_path / "run", seeds=[42], catches=[1], epsilons=[0.02])
    assert all(row["savings_mean"] is None for row in result["summary"])
    text = (tmp_path / "run/cascade-summary.json").read_text()
    assert "NaN" not in text
    assert json.loads(text)["summary"][0]["savings_mean"] is None


def test_gate_skips_ineligible_models_at_no_cost():
    eligible = np.zeros(OUTCOMES.shape, dtype=bool)
    eligible[0, 0] = True  # cheap only where it is right
    m = chain_metrics([0], OUTCOMES, COSTS, 2, catch=0.0, eligible=eligible)
    # task0: 1; tasks 1-3 go straight to the frontier: 10 each -> 31 vs 40
    assert m["savings"] == pytest.approx(1 - 31 / 40)
    assert m["degradation_rate"] == 0 and m["accuracy"] == 0.75
    assert m["frontier_call_rate"] == 0.75 and m["mean_calls"] == 1
    everything = chain_metrics([0], OUTCOMES, COSTS, 2, catch=0.0,
                               eligible=np.ones(OUTCOMES.shape, dtype=bool))
    assert everything == chain_metrics([0], OUTCOMES, COSTS, 2, catch=0.0)


def test_gate_thresholds_scores_and_rejects_bad_input():
    from tine.cascade import gate
    scores = np.array([[0.9, 0.2, 0.0], [0.5, 0.5, 0.0]])
    assert gate(scores, None) is None
    assert gate(scores, 0.5).tolist() == [[True, False, False], [True, True, False]]
    for bad in [(None, 0.5), (scores, 1.5), (scores * 2, 0.5), (scores[0], 0.5)]:
        with pytest.raises(ValueError):
            gate(*bad)
    with pytest.raises(ValueError):
        chain_metrics([0], OUTCOMES, COSTS, 2, eligible=OUTCOMES.astype(float))


def test_gated_certification_uses_the_gate_when_the_check_is_weak():
    from tine.cascade import certify_gated_chain
    n = 400
    outcomes = np.ones((n, 2), dtype=bool)
    outcomes[: n // 4, 0] = False              # cheap model wrong on a quarter
    costs = np.tile([1.0, 10.0], (n, 1))
    passes = np.ones((n, 2), dtype=bool)       # the check catches nothing
    scores = np.where(outcomes, 0.9, 0.1)      # the predictor knows
    part = (outcomes, costs, passes, scores)
    assert certify_gated_chain([[0]], (None,), part, part, 1, epsilon=0.02, delta=0.05) == ([], None, None)
    chain, tau, bound = certify_gated_chain([[0]], (None, 0.5), part, part, 1, epsilon=0.02, delta=0.05)
    assert chain == [0] and tau == 0.5 and bound <= 0.02
    with pytest.raises(ValueError):
        certify_gated_chain([[0]], (), part, part, 1, epsilon=0.02, delta=0.05)


def test_gated_experiment_reports_gated_method(tmp_path):
    frame = synthetic_fixture(n_tasks=400, seed=5)
    result = run_cascade_experiment(frame, "frontier", tmp_path, seeds=range(2), epsilons=(0.02,),
                                    catches=(0.9,), gate_method="logistic")
    runs = pd.read_csv(tmp_path / "cascade-runs.csv")
    assert set(runs.method) == {"uncertified", "certified", "gated_certified"}
    assert runs.loc[runs.method != "gated_certified", "gate_tau"].isna().all()
    assert result["gate"]["method"] == "logistic"
    assert "gated_certified" in (tmp_path / "report.md").read_text()
    # choose / certify / test exclude the tasks the predictor was fitted on
    assert runs.test_tasks.max() < 400 * 0.3
    with pytest.raises(ValueError):
        run_cascade_experiment(frame, "frontier", tmp_path, gate_method="oracle")
    with pytest.raises(ValueError):
        run_cascade_experiment(frame, "frontier", tmp_path, gate_method="knn", gate_thresholds=())


def test_gate_cli(tmp_path, capsys):
    from tine.data import write_outcomes
    write_outcomes(synthetic_fixture(n_tasks=300, seed=6), tmp_path / "o.jsonl")
    main(["cascade", "--input", str(tmp_path / "o.jsonl"), "--frontier", "frontier",
          "--output", str(tmp_path / "out"), "--seeds", "1", "--epsilon", "0.02", "--catch", "0.9",
          "--gate", "knn", "--gate-threshold", "0.7", "0.9"])
    assert "gated_certified" in capsys.readouterr().out


def test_per_model_catch_rates(tmp_path):
    rng = np.random.default_rng(0)
    passes = simulated_passes(np.zeros((4000, 2), dtype=bool), [1.0, 0.25], rng)
    assert not passes[:, 0].any() and 0.7 < passes[:, 1].mean() < 0.8
    with pytest.raises(ValueError):
        simulated_passes(np.zeros((5, 2), dtype=bool), [1.0], rng)
    frame = synthetic_fixture(n_tasks=200, seed=5)
    rates = {model: 1.0 for model in frame.model.unique() if model != "frontier"}
    result = run_cascade_experiment(frame, "frontier", tmp_path, seeds=range(2), epsilons=(0.02,),
                                    catches=(), catch_rates=rates)
    assert {row["catch"] for row in result["summary"]} == {"per-model"}
    assert all(row["degradation_max"] == 0 for row in result["summary"])
    with pytest.raises(ValueError, match="lack models"):
        run_cascade_experiment(frame, "frontier", tmp_path, catch_rates={"cheap": 0.5})
    pd.DataFrame({"model": list(rates), "catch_rate": 0.9}).to_csv(tmp_path / "rates.csv", index=False)
    from tine.data import write_outcomes
    write_outcomes(frame, tmp_path / "o.jsonl")
    main(["cascade", "--input", str(tmp_path / "o.jsonl"), "--frontier", "frontier", "--seeds", "1",
          "--output", str(tmp_path / "out"), "--epsilon", "0.02", "--catch", "1.0",
          "--catch-rates", str(tmp_path / "rates.csv")])
    assert set(pd.read_csv(tmp_path / "out" / "cascade-summary.csv")["catch"]) == {"1.0", "per-model"}


def test_gate_can_use_supplied_embeddings(tmp_path):
    from tine.pipeline import prompt_hash
    frame = synthetic_fixture(n_tasks=300, seed=5)
    tasks = frame.drop_duplicates(["tenant", "task_id"])
    vectors = tmp_path / "vectors.npz"
    np.savez(vectors, tenants=tasks.tenant.to_numpy(dtype=str), task_ids=tasks.task_id.to_numpy(dtype=str),
             prompt_hashes=[prompt_hash(prompt) for prompt in tasks.prompt],
             embeddings=np.random.default_rng(0).normal(size=(len(tasks), 8)))
    result = run_cascade_experiment(frame, "frontier", tmp_path / "out", seeds=[0], epsilons=(0.02,),
                                    catches=(0.9,), gate_method="knn", gate_embeddings=vectors)
    assert result["gate"]["features"] == "supplied_embeddings" and result["gate"]["embedding_sha256"]
    with pytest.raises(ValueError, match="requires a gate method"):
        run_cascade_experiment(frame, "frontier", tmp_path / "out", gate_embeddings=vectors)
