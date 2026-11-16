import json

import numpy as np
import pandas as pd
import pytest

from tine.data import synthetic_fixture
from tine.pipeline import complete_tasks, embedding_fingerprint, load_embeddings, preflight, prompt_hash, run_experiment, split_tasks


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


def test_loading_embeddings_retains_only_requested_tasks(tmp_path):
    tasks = pd.DataFrame([{"tenant": "a", "task_id": "x", "prompt": "hello"}])
    path = tmp_path / "features.npz"
    np.savez(path, tenants=["a", "a", "other"], task_ids=["x", "unused", "x"],
             prompt_hashes=[prompt_hash("hello"), prompt_hash("unused"), prompt_hash("private")],
             embeddings=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert set(load_embeddings(path, tasks)) == {("a", "x")}


def test_embedding_fingerprint_tracks_values_and_identity_not_archive_order():
    tasks = pd.DataFrame([
        {"tenant": "a", "task_id": "x", "prompt": "hello"},
        {"tenant": "a", "task_id": "y", "prompt": "world"},
    ])
    lookup = {("a", "x"): np.array([1.0, 2.0]), ("a", "y"): np.array([3.0, 4.0])}
    original = embedding_fingerprint(lookup, tasks)
    reordered = {key: value.astype(">f8") for key, value in reversed(list(lookup.items()))}
    reordered[("unused", "task")] = np.array([50.0, 60.0])
    assert embedding_fingerprint(reordered, tasks.iloc[::-1]) == original
    lookup[("a", "x")][0] += 0.25
    assert embedding_fingerprint(lookup, tasks) != original
    lookup[("a", "x")][0] -= 0.25
    tasks.loc[0, "prompt"] = "different prompt"
    assert embedding_fingerprint(lookup, tasks) != original


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


@pytest.mark.parametrize("fixed_embeddings", [False, True])
@pytest.mark.parametrize("rule,loss", [("absolute", "degradation"), ("relative", "net")])
def test_latency_cli_reports_each_tenant_and_matches_frozen_batch_decisions(tmp_path, fixed_embeddings, rule, loss):
    from tine.__main__ import main
    from tine.data import write_outcomes

    first = synthetic_fixture(40)
    second = first.copy()
    second["tenant"] = "second-team"
    source = tmp_path / "outcomes.jsonl"
    write_outcomes(pd.concat([first, second]), source)
    output = tmp_path / "report"
    command = ["run", "--input", str(source), "--frontier", "frontier", "--output", str(output)]
    command += ["--rule", rule, "--loss", loss]
    command += ["--epsilon", "0.04" if loss == "net" else "0.02"]
    if fixed_embeddings:
        tasks = pd.concat([first, second]).drop_duplicates(["tenant", "task_id"])
        vectors = tmp_path / "embeddings.npz"
        np.savez(vectors, tenants=tasks.tenant.to_numpy(dtype=str), task_ids=tasks.task_id.to_numpy(dtype=str),
                 prompt_hashes=tasks.prompt.map(prompt_hash).to_numpy(dtype=str),
                 embeddings=np.random.default_rng(17).normal(size=(len(tasks), 8)))
        command += ["--embeddings", str(vectors)]
    main([*command, "--latency-samples", "3"])
    summary = json.loads((output / "summary.json").read_text())
    if fixed_embeddings:
        assert len(summary["embedding_sha256"]) == 64
        assert summary["embedding_fingerprint_format"] == "tine-embedding-features-v1"
    else:
        assert summary["embedding_sha256"] is None
    report = json.loads((output / "routing-latency.json").read_text())
    assert report["synthetic"] is True
    assert report["rule"] == rule and report["loss"] == loss
    assert report["environment"]["machine"]
    assert "network" in report["scope"]
    assert len(report["results"]) == 6
    decisions = pd.read_csv(output / "decisions.csv").set_index(["tenant", "method", "task_id"])
    metrics = pd.read_csv(output / "metrics.csv").set_index(["tenant", "method"])
    for result in report["results"]:
        assert result["rule"] == rule
        key = (result["tenant"], result["method"])
        assert result["samples"] == metrics.loc[key, "routing_latency_samples"] == 3
        assert 0 <= result["p50_ms"] <= result["p99_ms"] <= result["max_ms"]
        for observation in result["observations"]:
            assert decisions.loc[(*key, observation["task_id"]), "model"] == observation["selected_model"]
            assert "prompt" not in observation
    # Disabling the optional benchmark must remove old measurements on rerun.
    main(command)
    assert not (output / "routing-latency.json").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["latency_samples_requested"] == 0
    assert all(item["routing_p99_ms"] is None for item in summary["metrics"])


@pytest.mark.parametrize("samples", [-1, True, 1.5])
def test_invalid_latency_count_cannot_replace_existing_output(tmp_path, samples):
    output = tmp_path / "run"
    output.mkdir()
    (output / "report.md").write_text("previous results")
    with pytest.raises(ValueError, match="latency_samples"):
        run_experiment(synthetic_fixture(20), "frontier", output, latency_samples=samples)
    assert (output / "report.md").read_text() == "previous results"


def test_relative_rule_and_net_loss_are_recorded_and_validated(tmp_path):
    frame = synthetic_fixture(n_tasks=400, seed=3)
    summary = run_experiment(frame, "frontier", tmp_path / "run", rule="relative", loss="net")
    assert summary["rule"] == "relative" and summary["loss"] == "net"
    calibration = json.loads((tmp_path / "run" / "calibration.json").read_text())
    assert all(item["loss"].startswith("net_accuracy_loss") and "gains" in item for item in calibration)
    assert all("difference of one-sided Clopper-Pearson bounds" in item["bound"] for item in calibration)
    report = (tmp_path / "run" / "report.md").read_text()
    assert "Rule: relative. Loss: net." in report and "Relative rule" in report and "Net loss" in report
    assert "Calibration bounds net accuracy loss" in report
    assert "Calibration bounds the event" not in report
    assert "each at half the corrected error level" in report
    default = run_experiment(frame, "frontier", tmp_path / "default")
    assert default["rule"] == "absolute" and default["loss"] == "degradation"
    assert "Calibration bounds the event" in (tmp_path / "default/report.md").read_text()
    for bad in [{"rule": "best"}, {"loss": "gross"}]:
        with pytest.raises(ValueError):
            run_experiment(frame, "frontier", tmp_path / "bad", **bad)
