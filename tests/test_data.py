import json

import numpy as np
import pandas as pd
import pytest

from tine.data import (
    import_llmrouterbench,
    read_outcomes,
    synthetic_fixture,
    validate_outcomes,
    write_outcomes,
)


def test_fixture_is_reproducible_and_has_complete_model_outcomes():
    frame = synthetic_fixture(50, seed=7)
    pd.testing.assert_frame_equal(frame, synthetic_fixture(50, seed=7))
    assert len(frame) == 150
    assert frame.groupby("task_id")["model"].nunique().eq(3).all()
    assert set(frame["cost_source"]) == {"synthetic"}


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_roundtrip_keeps_unknown_outcomes_and_false_labels(tmp_path, suffix):
    frame = synthetic_fixture(4)
    frame.loc[0, "success"] = None
    frame.loc[1, "success"] = False
    path = tmp_path / ("outcomes" + suffix)
    write_outcomes(frame, path)
    loaded = read_outcomes(path)
    assert pd.isna(loaded.loc[0, "success"])
    assert loaded.loc[1, "success"] is False
    with pytest.raises(ValueError, match="Pending"):
        read_outcomes(path, require_known=True)


@pytest.mark.parametrize("suffix", [".csv", ".jsonl", ".ndjson"])
def test_roundtrip_preserves_recorded_costs_and_measurements(tmp_path, suffix):
    frame = synthetic_fixture(2)
    frame["cost_usd"] = [1.234567890123456e-8, 1.234567890123456e-11, 1.234567890123456e-18] * 2
    frame["latency_ms"] = [0.1234567890123456, np.nan, 12345.67890123456] * 2
    frame["input_tokens"] = [123456789012345, 456, 789] * 2
    path = tmp_path / ("outcomes" + suffix)
    write_outcomes(frame, path)
    loaded = read_outcomes(path)
    np.testing.assert_array_equal(loaded.cost_usd, frame.cost_usd)
    np.testing.assert_array_equal(loaded.latency_ms, frame.latency_ms)
    np.testing.assert_array_equal(loaded.input_tokens, frame.input_tokens)


@pytest.mark.parametrize("text", ["NA", "N/A", "null", "NULL", "NaN", "None", "001", "123", "true", 'Explain "λ",\nthen return 1'])
def test_csv_roundtrip_preserves_text_fields(tmp_path, text):
    frame = synthetic_fixture(1).iloc[[0]].copy()
    text_columns = ["tenant", "task_id", "prompt", "task_type", "model"]
    for column in text_columns:
        frame[column] = text
    frame["success"] = None
    frame["input_tokens"] = np.nan
    frame["timestamp"] = None
    path = tmp_path / "outcomes.csv"
    write_outcomes(frame, path)
    loaded = read_outcomes(path)
    assert loaded[text_columns].iloc[0].tolist() == [text] * len(text_columns)
    assert loaded[["success", "input_tokens", "timestamp"]].isna().all().all()


def test_jsonl_roundtrip_preserves_timestamps_and_nullable_provenance(tmp_path):
    frame = synthetic_fixture(1).iloc[[0]].copy()
    timestamp = pd.Timestamp("2026-11-07T09:00:00.123456+00:00")
    frame["timestamp"] = timestamp
    frame["source_metadata"] = [{
        "score": np.float64(0.1234567890123456),
        "attempts": np.int64(2),
        "checks": [np.bool_(True), None, np.nan, pd.NA],
        "measurements": np.array([0.1234567890123456, np.nan]),
    }]
    path = tmp_path / "outcomes.jsonl"
    write_outcomes(frame, path)
    loaded = read_outcomes(path)
    assert pd.Timestamp(loaded.loc[0, "timestamp"]) == timestamp
    assert loaded.loc[0, "source_metadata"] == {
        "score": 0.1234567890123456, "attempts": 2,
        "checks": [True, None, None, None],
        "measurements": [0.1234567890123456, None],
    }
    assert "NaN" not in path.read_text()


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_json_lines_preserve_unicode_separators_inside_records(tmp_path, suffix):
    frame = synthetic_fixture(1)
    text = "First line\u2028second line\u2029next paragraph\u0085last line"
    frame["prompt"] = text
    frame["source_metadata"] = [{"note": text} for _ in range(len(frame))]
    path = tmp_path / ("outcomes" + suffix)
    write_outcomes(frame, path)

    loaded = read_outcomes(path)

    assert loaded["prompt"].tolist() == [text] * len(frame)
    assert loaded["source_metadata"].tolist() == frame["source_metadata"].tolist()


@pytest.mark.parametrize("column,value", [
    ("cost_usd", -1), ("cost_usd", np.inf), ("cost_usd", np.nan),
    ("input_tokens", 1.5), ("cached_tokens", 10000), ("latency_ms", -1),
    ("success", "ungraded"), ("success", 0.5), ("prompt", ""),
])
def test_invalid_measurements_are_rejected(column, value):
    frame = synthetic_fixture(2)
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_outcomes(frame)


def test_duplicate_model_outcome_and_conflicting_prompts_are_rejected():
    frame = synthetic_fixture(2)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_outcomes(pd.concat([frame, frame.iloc[[0]]]))
    frame.loc[0, "prompt"] = "different request"
    with pytest.raises(ValueError, match="Conflicting prompt"):
        validate_outcomes(frame)


@pytest.mark.parametrize("column", ["cost_usd", "input_tokens", "output_tokens", "cached_tokens", "latency_ms"])
@pytest.mark.parametrize("value", [False, True])
def test_import_rejects_boolean_measurements(tmp_path, column, value):
    records = synthetic_fixture(2).to_dict(orient="records")
    records[0][column] = value
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match=column):
        read_outcomes(path)


def test_nullable_numpy_measurements_keep_unknowns_and_reject_booleans():
    frame = synthetic_fixture(2)
    frame["input_tokens"] = pd.Series([pd.NA, 100, 100, 100, 100, 100], dtype="Int64")
    assert pd.isna(validate_outcomes(frame).loc[0, "input_tokens"])
    frame["input_tokens"] = frame["input_tokens"].astype(object)
    frame.loc[1, "input_tokens"] = np.bool_(False)
    with pytest.raises(ValueError, match="input_tokens"):
        validate_outcomes(frame)


@pytest.mark.parametrize("column", ["cost_usd", "input_tokens", "output_tokens", "cached_tokens", "latency_ms"])
def test_complex_measurements_are_not_silently_converted_to_real(column):
    frame = synthetic_fixture(2)
    frame[column] = frame[column].astype(complex) + 2j
    with pytest.raises(ValueError, match=column):
        validate_outcomes(frame)


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_failed_outcome_encoding_preserves_existing_file(tmp_path, suffix):
    path = tmp_path / ("outcomes" + suffix)
    frame = synthetic_fixture(2)
    write_outcomes(frame, path)
    previous = path.read_bytes()
    frame["prompt"] = "invalid\ud800"
    with pytest.raises(UnicodeEncodeError):
        write_outcomes(frame, path)
    assert path.read_bytes() == previous


def test_failed_outcome_publication_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "outcomes.jsonl"
    frame = synthetic_fixture(2)
    write_outcomes(frame, path)
    previous = path.read_bytes()

    def fail_replace(*args):
        raise OSError("disk failure")

    monkeypatch.setattr("tine.artifacts.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        write_outcomes(synthetic_fixture(3), path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def _result_file(root, *, score=1, cost=0.01, dataset="humaneval", model="small", run="20260101_000000"):
    path = root / dataset / "test" / model / f"{run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset, "split": "test", "model_name": model,
        "cost": 99.0,
        "records": [{"index": 0, "origin_query": "Implement add(a, b)", "prompt": "model-specific wrapper",
                     "score": score, "cost": cost, "prompt_tokens": 12, "completion_tokens": 8}],
    }
    path.write_text(json.dumps(payload))
    return path


def test_upstream_adapter_preserves_costs_and_alignment_without_inventing_measurements(tmp_path):
    _result_file(tmp_path, model="small")
    _result_file(tmp_path, model="large", cost=0.02)
    _result_file(tmp_path, dataset="mbpp", model="large", cost=0.03)
    frame = import_llmrouterbench(tmp_path)
    assert set(frame["task_id"]) == {"humaneval:test:0", "mbpp:test:0"}
    assert frame["prompt"].eq("Implement add(a, b)").all()
    assert sorted(frame["cost_usd"]) == [0.01, 0.02, 0.03]
    assert frame["cached_tokens"].isna().all()
    assert frame["latency_ms"].isna().all()
    assert frame["source_sha256"].str.len().eq(64).all()


def test_fractional_score_needs_explicit_label_policy(tmp_path):
    path = _result_file(tmp_path, score=0.8)
    with pytest.raises(ValueError, match="success must"):
        import_llmrouterbench(path)
    frame = import_llmrouterbench(path, score_threshold=0.9)
    assert frame.loc[0, "success"] is False
    assert frame.loc[0, "source_score"] == 0.8


def test_null_label_is_pending_and_missing_cost_never_uses_aggregate(tmp_path):
    path = _result_file(tmp_path, score=None)
    assert import_llmrouterbench(path)["success"].isna().all()
    _result_file(tmp_path, score=1, cost=None)
    with pytest.raises(ValueError, match="per-record cost"):
        import_llmrouterbench(path)


def test_multiple_runs_require_explicit_latest_selection(tmp_path):
    _result_file(tmp_path, score=0)
    _result_file(tmp_path, score=1, run="20260201_000000")
    with pytest.raises(ValueError, match="Multiple recorded runs"):
        import_llmrouterbench(tmp_path)
    frame = import_llmrouterbench(tmp_path, run_selection="latest")
    assert len(frame) == 1 and frame.loc[0, "success"] is True


def _outcomes(dataset, models, n_tasks, cost=0.01):
    return pd.DataFrame([
        {"tenant": "benchmark", "task_id": f"{dataset}:test:{i}", "prompt": f"{dataset} task {i}",
         "task_type": dataset, "model": model, "success": (i + j) % 2 == 0,
         "cost_usd": 0.0 if model == "unpriced" else cost}
        for i in range(n_tasks) for j, model in enumerate(models)
    ])


def test_combine_outcomes_pools_groups_and_drops_unusable_models(tmp_path):
    from tine.__main__ import main
    from tine.data import combine_outcomes

    first = _outcomes("humaneval", ["big", "small", "unpriced", "only-here"], 3)
    second = _outcomes("mbpp", ["big", "small", "unpriced"], 2)
    frame, report = combine_outcomes([first, second], task_type="coding")
    assert set(frame.task_type) == {"coding"}
    assert set(frame.source_task_type) == {"humaneval", "mbpp"}
    assert report["tasks_by_source"] == {"humaneval": 3, "mbpp": 2}
    assert report["models_kept"] == ["big", "small"]
    assert report["models_dropped"] == {"only-here": "missing_from_some_inputs",
                                        "unpriced": "cost_not_recorded_all_zero"}
    kept, _ = combine_outcomes([first, second], keep_unpriced_models=True, exclude_models=["small"])
    assert sorted(kept.model.unique()) == ["big", "unpriced"]
    assert set(kept.task_type) == {"humaneval", "mbpp"}

    paths = []
    for name, part in (("a.jsonl", first), ("b.jsonl", second)):
        write_outcomes(part, tmp_path / name)
        paths.append(str(tmp_path / name))
    main(["combine-outcomes", "--input", *paths, "--output", str(tmp_path / "c.jsonl"),
          "--task-type", "coding"])
    assert len(read_outcomes(tmp_path / "c.jsonl")) == 5 * 2
