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
