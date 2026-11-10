import numpy as np
import pandas as pd
import pytest

from tine.data import synthetic_fixture
from tine.latency import benchmark_routing
from tine.routing import Router, TASK_COLUMNS


@pytest.fixture
def setup():
    frame = synthetic_fixture(40)
    ids = frame.task_id.unique()
    router = Router("frontier").fit(
        frame.loc[frame.task_id.isin(ids[:20])], frame.loc[frame.task_id.isin(ids[20:30])],
    )
    tasks = frame.loc[frame.task_id.isin(ids[30:])].drop_duplicates("task_id")
    return router, tasks


def test_timings_exclude_warmup_and_preserve_explicit_frontier_policy(setup, monkeypatch):
    router, tasks = setup
    calls = []

    def scores(task, method):
        assert list(task.columns) == TASK_COLUMNS
        assert len(task) == 1
        calls.append(task.task_id.iloc[0])
        # A score of one would qualify the cheapest model without force_frontier.
        return np.ones((1, len(router.model_names)))

    monkeypatch.setattr(router, "predict_scores", scores)
    clock = iter([0, 1_000_000, 2_000_000, 4_000_000, 5_000_000, 8_000_000])
    monkeypatch.setattr("tine.latency.time.perf_counter_ns", lambda: next(clock))
    result = benchmark_routing(router, tasks, {"coding": {"tau": None, "force_frontier": True}},
                               "logistic", samples=3)
    assert len(calls) == 6
    assert result["warmup_calls"] == result["samples"] == 3
    assert result["p50_ms"] == 2
    assert result["p99_ms"] == pytest.approx(2.98)
    assert result["max_ms"] == 3
    assert {item["selected_model"] for item in result["observations"]} == {"frontier"}
    assert len({item["task_id"] for item in result["observations"]}) == 3


def test_latency_sampling_is_bounded_reproducible_and_ignores_outcomes(setup):
    router, tasks = setup
    policies = {"coding": {"tau": 0.0, "force_frontier": False}}
    original = tasks.copy(deep=True)
    first = benchmark_routing(router, tasks, policies, "knn", samples=100, seed=7)
    changed = tasks.sample(frac=1, random_state=5)
    changed["success"] = None
    changed["cost_usd"] = 9999
    second = benchmark_routing(router, changed, policies, "knn", samples=100, seed=7)
    assert first["samples"] == len(tasks)
    assert first["requested_samples"] == 100
    def key(result):
        return [(item["task_id"], item["selected_model"]) for item in result["observations"]]
    assert key(first) == key(second)
    assert {model for _, model in key(first)} == {"cheap"}
    pd.testing.assert_frame_equal(tasks, original)


@pytest.mark.parametrize("samples", [0, -1, True])
def test_latency_requires_positive_sample_limit(setup, samples):
    router, tasks = setup
    with pytest.raises(ValueError, match="positive integer"):
        benchmark_routing(router, tasks, {}, "logistic", samples=samples)


def test_latency_requires_tenant_isolation_and_policy_coverage(setup):
    router, tasks = setup
    with pytest.raises(ValueError, match="frozen calibration policy"):
        benchmark_routing(router, tasks, {}, "cluster", samples=1)
    tasks.loc[tasks.index[0], "tenant"] = "another-tenant"
    with pytest.raises(ValueError, match="tenant"):
        benchmark_routing(router, tasks, {"coding": {"tau": 0, "force_frontier": False}},
                          "cluster", samples=1)
