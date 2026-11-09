import json

import numpy as np
import pandas as pd
import pytest

from tine.data import synthetic_fixture
from tine.pipeline import complete_tasks, load_embeddings, preflight, prompt_hash, run_experiment, split_tasks


def test_duplicate_prompts_stay_in_same_partition():
    tasks = pd.DataFrame({"tenant": ["a"] * 20, "task_id": [str(i) for i in range(20)],
                          "prompt": [f"question {i % 10}" for i in range(20)], "task_type": "coding"})
    split = split_tasks(tasks)
    assert set(split.split) == {"train", "score_calibration", "risk_calibration", "test"}
    assert split.groupby("prompt").split.nunique().eq(1).all()
    pd.testing.assert_frame_equal(split, split_tasks(tasks))


def test_incomplete_outcome_is_excluded_not_filled():
    frame = synthetic_fixture(n_tasks=30)
    missing_id = frame.iloc[0].task_id
    frame = frame.drop(frame.index[0])
    result, tasks, excluded = complete_tasks(frame, "frontier")
    assert missing_id not in tasks.task_id.to_list()
    assert missing_id not in result.task_id.to_list()
    assert len(excluded) == 1


def test_duplicate_tasks_cannot_inflate_calibration_sample_size():
    frame = synthetic_fixture(n_tasks=30)
    copy = frame.loc[frame.task_id == "synthetic-0"].copy()
    copy["task_id"] = "copy-0"
    _, tasks, excluded = complete_tasks(pd.concat([frame, copy], ignore_index=True), "frontier")
    assert len(tasks) == 30
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "duplicate_prompt"


def test_embeddings_are_bound_to_prompts_and_no_pickle(tmp_path):
    tasks = pd.DataFrame([{"tenant": "a", "task_id": "x", "prompt": "hello"}])
    path = tmp_path / "features.npz"
    np.savez(path, tenants=["a"], task_ids=["x"], prompt_hashes=[prompt_hash("hello")], embeddings=[[1.0, 2.0]])
    lookup = load_embeddings(path, tasks)
    assert np.allclose(lookup[("a", "x")], [1, 2])
    tasks.loc[0, "prompt"] = "changed"
    with pytest.raises(ValueError, match="stale"):
        load_embeddings(path, tasks)


@pytest.mark.parametrize("vectors", [[[1+2j, 3+4j]], [["one", "two"]], [[True, False]]])
def test_embeddings_require_real_numeric_vectors(tmp_path, vectors):
    tasks = pd.DataFrame([{"tenant": "a", "task_id": "x", "prompt": "hello"}])
    path = tmp_path / "features.npz"
    np.savez(path, tenants=["a"], task_ids=["x"], prompt_hashes=[prompt_hash("hello")], embeddings=vectors)
    with pytest.raises(ValueError, match="real numeric"):
        load_embeddings(path, tasks)


def test_embedding_metadata_shape_is_checked_before_lookup(tmp_path):
    tasks = pd.DataFrame([{"tenant": "a", "task_id": "x", "prompt": "hello"}])
    path = tmp_path / "features.npz"
    np.savez(path, tenants=[["a"]], task_ids=["x"], prompt_hashes=[prompt_hash("hello")], embeddings=[[1, 2]])
    with pytest.raises(ValueError, match="one-dimensional"):
        load_embeddings(path, tasks)


def test_full_experiment_reports_fallback_and_no_split_leakage(tmp_path):
    frame = synthetic_fixture(n_tasks=60, seed=12)
    result = run_experiment(frame, "frontier", tmp_path, synthetic=True)
    assert result["synthetic"] is True
    assert result["complete_tasks"] == 60
    assert {item["method"] for item in result["metrics"]} == {"frontier", "random", "oracle", "logistic", "knn", "cluster"}
    calibration = json.loads((tmp_path / "calibration.json").read_text())
    assert all(item["force_frontier"] for item in calibration)
    readiness = preflight(frame, "frontier")
    # One group x three methods; thresholds use fixed-sequence testing, not Bonferroni.
    assert readiness["comparisons"] == len(calibration) == 3
    assert readiness["minimum_risk_tasks_at_zero_degradations"] == 203
    assert readiness["groups"][0]["risk_calibration"] == calibration[0]["n"]
    assert readiness["groups"][0]["enough_risk_tasks_at_zero_degradations"] is False
    choices = pd.read_csv(tmp_path / "decisions.csv")
    assert choices.loc[choices.method.isin(["logistic", "knn", "cluster"]), "model"].eq("frontier").all()
    splits = pd.read_csv(tmp_path / "splits.csv")
    assert splits.groupby("prompt_hash").split.nunique().eq(1).all()
    assert (tmp_path / "cost-quality.png").read_bytes().startswith(b"\x89PNG")
    assert "SYNTHETIC" in (tmp_path / "report.md").read_text()



def test_dataset_fingerprint_distinguishes_small_costs(tmp_path):
    frame = synthetic_fixture(n_tasks=16)
    frame.loc[0, "cost_usd"] = 1e-11
    first = run_experiment(frame, "frontier", tmp_path / "first", thresholds=[0.5])
    frame.loc[0, "cost_usd"] = 2e-11
    second = run_experiment(frame, "frontier", tmp_path / "second", thresholds=[0.5])
    assert first["dataset_sha256"] != second["dataset_sha256"]


def test_empty_threshold_grid_writes_frontier_fallback_report(tmp_path):
    frame = synthetic_fixture(n_tasks=60, seed=12)
    run_experiment(frame, "frontier", tmp_path, thresholds=[], synthetic=True)

    calibration = json.loads((tmp_path / "calibration.json").read_text())
    assert all(item["force_frontier"] and item["thresholds_tested"] == 0 for item in calibration)
    choices = pd.read_csv(tmp_path / "decisions.csv")
    assert choices.loc[choices.method.isin(["logistic", "knn", "cluster"]), "model"].eq("frontier").all()
    curves = pd.read_csv(tmp_path / "curves.csv")
    assert curves.empty
    assert {"tenant", "method", "tau", "mean_cost_usd", "accuracy"} <= set(curves.columns)
    assert (tmp_path / "cost-quality.png").is_file()


def test_insufficient_unique_prompts_refused():
    tasks = pd.DataFrame({"tenant": "x", "task_id": [str(i) for i in range(7)], "prompt": [str(i) for i in range(7)]})
    with pytest.raises(ValueError, match="at least 8"):
        split_tasks(tasks)


def test_source_split_provenance_does_not_override_experiment_partitions(tmp_path):
    frame = synthetic_fixture(n_tasks=30)
    frame["split"] = "upstream-test"
    frame["split_source"] = "upstream-release"
    original = frame.copy(deep=True)
    result = run_experiment(frame, "frontier", tmp_path, thresholds=[0.5])
    pd.testing.assert_frame_equal(frame, original)
    partitions = pd.read_csv(tmp_path / "splits.csv")
    assert set(partitions.split) == {"train", "score_calibration", "risk_calibration", "test"}
    assert result["complete_tasks"] == 30


def test_failed_demo_rerun_preserves_report_and_source_snapshot(tmp_path, monkeypatch):
    from tine.__main__ import main

    output = tmp_path / "demo"
    main(["demo", "--tasks", "20", "--output", str(output)])
    previous = {path.name: path.read_bytes() for path in output.iterdir()}

    def fail_plot(*args):
        raise OSError("plot failed")

    monkeypatch.setattr("tine.pipeline._plot", fail_plot)
    with pytest.raises(OSError, match="plot failed"):
        main(["demo", "--tasks", "25", "--output", str(output)])
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous


def test_failed_chart_write_closes_figure_and_leaves_no_new_run(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    figures = plt.get_fignums()

    def fail_save(*args, **kwargs):
        raise OSError("chart storage failure")

    monkeypatch.setattr(Figure, "savefig", fail_save)
    with pytest.raises(OSError, match="chart storage failure"):
        run_experiment(synthetic_fixture(20), "frontier", tmp_path / "run", thresholds=[0.5])
    assert plt.get_fignums() == figures
    assert not (tmp_path / "run").exists()


def test_run_cli_recognizes_synthetic_provenance_in_mixed_data(tmp_path):
    from tine.__main__ import main
    from tine.data import write_outcomes

    frame = synthetic_fixture(20)
    frame["cost_source"] = "recorded"
    frame["success_label_source"] = "graded"
    frame.loc[0, "cost_source"] = "synthetic"
    source = tmp_path / "mixed.jsonl"
    write_outcomes(frame, source)
    output = tmp_path / "run"
    main(["run", "--input", str(source), "--frontier", "frontier", "--output", str(output)])
    assert json.loads((output / "summary.json").read_text())["synthetic"] is True
    assert "SYNTHETIC" in (output / "report.md").read_text()
    assert not (output / "synthetic-outcomes.jsonl").exists()
