import numpy as np
import pandas as pd
import pytest

from tine.routing import Router, route_indices


def outcomes(prefix, n=20, tenant="team-a"):
    rows = []
    for i in range(n):
        for model, cost in [("cheap", 0.01), ("frontier", 0.1)]:
            rows.append({
                "tenant": tenant, "task_id": f"{prefix}-{i}",
                "prompt": f"{prefix} {'simple addition' if i % 2 else 'complex graph proof'} example {i}",
                "task_type": "coding", "model": model,
                "success": model == "frontier" or bool(i % 2), "cost_usd": cost,
            })
    return pd.DataFrame(rows)


def tasks(frame):
    return frame[["tenant", "task_id", "prompt", "task_type"]].drop_duplicates()


@pytest.fixture
def fitted():
    return Router("frontier", n_neighbors=3, n_clusters=3).fit(outcomes("train"), outcomes("score"))


def test_cost_routing_and_explicit_fallback():
    scores = np.array([[0.9, 0.8, 0.1], [0.7, 0.8, 0.1], [1.0, 1.0, 0.0]])
    costs = np.array([[1.0, 2.0, 10.0]] * 3)
    assert route_indices(scores, costs, 0.85, 2).tolist() == [0, 2, 0]
    assert route_indices(scores, costs, 1.0, 2, force_frontier=True).tolist() == [2, 2, 2]
    assert route_indices(np.ones((1, 3)), np.ones((1, 3)), 0.5, 2).tolist() == [0]


@pytest.mark.parametrize("scores,costs,tau,index", [
    ([[np.nan, 1]], [[1, 2]], 0.5, 1),
    ([[1.1, 1]], [[1, 2]], 0.5, 1),
    ([[1, 1]], [[-1, 2]], 0.5, 1),
    ([[1, 1]], [[1]], 0.5, 1),
    ([[1, 1]], [[1, 2]], 1.1, 1),
    ([[1, 1]], [[1, 2]], 0.5, 2),
])
def test_invalid_decision_inputs(scores, costs, tau, index):
    with pytest.raises(ValueError):
        route_indices(scores, costs, tau, index)


@pytest.mark.parametrize("method", ["logistic", "knn", "cluster"])
def test_scores_are_probabilities_and_frontier_is_calibrated(fitted, method):
    result = fitted.predict_scores(tasks(outcomes("test")), method)
    assert result.shape == (20, 2)
    assert np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()
    assert np.all(result[:, fitted.frontier_index] == 1)


def test_cost_estimates_ignore_prediction_outcomes_and_calibration_costs():
    calibration = outcomes("score")
    calibration["cost_usd"] = 9999
    router = Router("frontier").fit(outcomes("train"), calibration)
    test = outcomes("test")
    test["cost_usd"] = 8888
    assert np.allclose(router.estimated_costs(tasks(test)), [[0.01, 0.1]] * 20)
    altered = tasks(test)
    altered["task_type"] = "new-task-type"
    assert np.allclose(router.estimated_costs(altered), [[0.01, 0.1]] * 20)


@pytest.mark.parametrize("partition", ["train", "score_calibration"])
@pytest.mark.parametrize("value", [False, np.bool_(True)])
def test_fit_rejects_boolean_costs(partition, value):
    frames = {"train": outcomes("train"), "score_calibration": outcomes("score")}
    frames[partition]["cost_usd"] = frames[partition]["cost_usd"].astype(object)
    frames[partition].loc[0, "cost_usd"] = value
    with pytest.raises(ValueError, match="cost_usd"):
        Router("frontier").fit(**frames)


def test_tenant_boundaries(fitted):
    with pytest.raises(ValueError, match="tenant"):
        fitted.predict_scores(tasks(outcomes("test", tenant="team-b")))
    with pytest.raises(ValueError, match="tenant"):
        fitted.estimated_costs(tasks(outcomes("test", tenant="team-b")))
    with pytest.raises(ValueError, match="tenant"):
        Router("frontier").fit(pd.concat([outcomes("a"), outcomes("b", tenant="team-b")]), outcomes("score"))
    with pytest.raises(ValueError, match="tenant"):
        Router("frontier").fit(outcomes("train"), outcomes("score", tenant="team-b"))


def test_task_and_prompt_leakage_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        Router("frontier").fit(outcomes("same"), outcomes("same"))
    train = outcomes("train")
    calibration = outcomes("score")
    calibration.loc[calibration["task_id"] == "score-0", "prompt"] = train.iloc[0]["prompt"]
    with pytest.raises(ValueError, match="disjoint"):
        Router("frontier").fit(train, calibration)


def test_unknown_labels_are_not_filled():
    train = outcomes("train")
    train["success"] = train["success"].astype("boolean")
    train.loc[train["model"] == "cheap", "success"] = None
    with pytest.raises(ValueError, match="no known training outcomes"):
        Router("frontier").fit(train, outcomes("score"))
    calibration = outcomes("score")
    calibration["success"] = calibration["success"].astype("boolean")
    calibration.loc[calibration["model"] == "cheap", "success"] = None
    with pytest.raises(ValueError, match="no known score-calibration outcomes"):
        Router("frontier").fit(outcomes("train"), calibration)


def test_constant_labels_and_tiny_training_sets():
    train, calibration = outcomes("train", n=1), outcomes("score", n=1)
    router = Router("frontier").fit(train, calibration)
    result = router.predict_scores(tasks(outcomes("test", n=1)))
    assert result.tolist() == [[0.0, 1.0]]


def test_fixed_embeddings_are_used_without_tfidf():
    train, calibration, test = outcomes("train"), outcomes("score"), outcomes("test")
    all_tasks = tasks(pd.concat([train, calibration, test]))
    lookup = {(row.tenant, row.task_id): np.array([1.0, i % 2, i / 100]) for i, row in enumerate(all_tasks.itertuples())}
    router = Router("frontier", embedding_lookup=lookup).fit(train, calibration)
    assert not hasattr(router, "vectorizer_")
    assert router.predict_scores(tasks(test)).shape == (20, 2)
    missing = tasks(outcomes("missing", n=1))
    with pytest.raises(ValueError, match="Missing fixed embeddings"):
        router.predict_scores(missing)


@pytest.mark.parametrize("lookup", [
    {}, {("a", "1"): [np.nan]}, {("a", "1"): [[1, 2]]},
    {("a", "1"): [1], ("a", "2"): [1, 2]},
    {("a", "1"): np.array([1+2j, 3+4j])},
])
def test_invalid_embeddings(lookup):
    with pytest.raises(ValueError):
        Router("frontier", embedding_lookup=lookup)


def test_frontier_and_duplicate_validation():
    with pytest.raises(ValueError, match="Frontier"):
        Router("absent").fit(outcomes("train"), outcomes("score"))
    with pytest.raises(ValueError, match="Duplicate"):
        Router("frontier").fit(pd.concat([outcomes("train"), outcomes("train")]), outcomes("score"))
