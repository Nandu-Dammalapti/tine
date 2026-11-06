import json

import numpy as np
import pandas as pd
import pytest

from tine.data import RoutingTable, load_llmrouterbench, make_synthetic_results, split_indices
from tine.features import TfidfEmbedder, load_embeddings, save_embeddings
from tine.schema import RESULT_COLUMNS, validate_results


def _row(**overrides):
    row = dict(tenant="t1", task_id="a", prompt="p", task_type="coding", model="m1",
               success=1, cost_usd=0.01, input_tokens=10, output_tokens=5, cached_tokens=0,
               latency_ms=1.0, timestamp="2026-11-06")
    row.update(overrides)
    return row


def test_pending_outcome_is_unknown_not_failure():
    frame = pd.DataFrame([_row(success="pending"), _row(model="m2", success=False)])
    out = validate_results(frame)
    assert np.isnan(out["success"].iloc[0])
    assert out["success"].iloc[1] == 0.0


def test_success_values_read_from_text():
    frame = pd.DataFrame([_row(model=f"m{i}", success=v)
                          for i, v in enumerate(["1.0", "0", "TRUE", "failure", " ", True])])
    np.testing.assert_array_equal(validate_results(frame)["success"].to_numpy(),
                                  [1, 0, 1, 0, np.nan, 1])
    with pytest.raises(ValueError):
        validate_results(pd.DataFrame([_row(success="0.5")]))


def test_rejects_missing_columns_and_duplicates():
    with pytest.raises(ValueError, match="Missing"):
        validate_results(pd.DataFrame([{"tenant": "t1"}]))
    with pytest.raises(ValueError, match="One row per"):
        validate_results(pd.DataFrame([_row(), _row()]))


def test_table_is_single_tenant():
    frame = pd.DataFrame([_row(), _row(tenant="t2")])
    with pytest.raises(ValueError, match="several tenants"):
        RoutingTable.from_results(frame)
    table = RoutingTable.from_results(frame, tenant="t2")
    assert table.tenant == "t2" and len(table) == 1


def test_table_pivot_and_complete():
    frame = pd.DataFrame([
        _row(task_id="a", model="m1", success=1),
        _row(task_id="a", model="m2", success=0),
        _row(task_id="b", model="m1", success="pending"),
        _row(task_id="b", model="m2", success=1),
    ])
    table = RoutingTable.from_results(frame)
    assert table.models == ["m1", "m2"]
    assert table.success.shape == (2, 2)
    complete = table.complete()
    assert list(complete.task_ids) == ["a"]


def test_split_is_disjoint_and_stratified():
    types = np.array(["x"] * 100 + ["y"] * 50)
    train, cal, test = split_indices(types, seed=1)
    assert len(set(train) | set(cal) | set(test)) == 150
    assert not set(train) & set(cal) and not set(cal) & set(test)
    assert (types[cal] == "y").sum() == 10


def test_synthetic_has_schema():
    frame = make_synthetic_results(n_tasks=20)
    assert tuple(frame.columns) == RESULT_COLUMNS
    assert len(frame) == 20 * 4


def test_llmrouterbench_layout(tmp_path):
    for model, scores in {"small": [1.0, 0.0], "big": [1.0, 1.0]}.items():
        folder = tmp_path / "humaneval" / "test" / model
        folder.mkdir(parents=True)
        records = [{"index": i, "prompt": f"q{i}", "origin_query": f"q{i}", "score": s,
                    "prompt_tokens": 10, "completion_tokens": 20, "cost": 0.001 * (i + 1)}
                   for i, s in enumerate(scores)]
        (folder / "20250101.json").write_text(json.dumps({"records": [{"score": 0.0}]}))
        (folder / "20250201.json").write_text(json.dumps({"performance": 1, "records": records}))
    frame = load_llmrouterbench(tmp_path)
    table = RoutingTable.from_results(frame)
    assert table.models == ["big", "small"]
    assert list(table.task_ids) == ["humaneval/test/0", "humaneval/test/1"]
    np.testing.assert_array_equal(table.success, [[1, 1], [1, 0]])  # newest run wins


def test_embeddings_roundtrip_in_requested_order(tmp_path):
    path = tmp_path / "e.npz"
    save_embeddings(path, ["a", "b", "c"], np.eye(3), "test-model")
    vectors = load_embeddings(path, ["c", "a"])
    np.testing.assert_allclose(vectors, [[0, 0, 1], [1, 0, 0]])
    with pytest.raises(ValueError, match="no embedding"):
        load_embeddings(path, ["zzz"])


def test_tfidf_embedder_unit_vectors():
    prompts = np.array(["fix the python bug", "solve this integral", "rewrite the email"] * 5)
    X = TfidfEmbedder(n_components=4).fit(prompts).transform(prompts)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
