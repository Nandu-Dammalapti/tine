import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "tine_llmrouter_baseline", Path(__file__).parents[1] / "scripts" / "llmrouter_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_reward_is_the_raw_outcome_at_weight_one_and_charges_cost_below_it():
    success = np.array([[True, True], [False, True]])
    cost = np.array([[1.0, 5.0], [1.0, 5.0]])
    assert baseline.reward(success, cost, 1.0).tolist() == [[1.0, 1.0], [0.0, 1.0]]
    mixed = baseline.reward(success, cost, 0.7)
    assert mixed[0, 0] > mixed[0, 1]  # both right: the cheaper model scores higher
    assert mixed[1, 1] > mixed[1, 0]  # only the dearer one is right


def test_training_rows_follow_the_given_model_order_within_a_query():
    rows = baseline.training_rows(np.array(["a", "b"]), np.array([[True, False], [True, True]]),
                                  np.ones((2, 2)), ["cheap", "frontier"], [1, 0], 1.0)
    assert [(row["query"], row["model_name"], row["embedding_id"]) for row in rows] == [
        ("a", "frontier", 0), ("a", "cheap", 0), ("b", "frontier", 1), ("b", "cheap", 1)]
    assert rows[0]["performance"] == 0.0 and rows[1]["performance"] == 1.0


def test_certified_weight_stops_at_the_first_failure_and_never_takes_a_loss():
    weights = [1.0, 0.8, 0.6, 0.4]
    savings = {1.0: 0.0, 0.8: 0.2, 0.6: 0.5, 0.4: 0.6}
    # 0.6 fails its bound, so 0.4 is never reached even though it would pass.
    counts = {1.0: 0, 0.8: 0, 0.6: 30, 0.4: 0}
    assert baseline.certified_weight(weights, counts, 300, savings, 0.02, 0.05) == 0.8
    assert baseline.certified_weight(weights, {**counts, 1.0: 30}, 300, savings, 0.02, 0.05) is None
    losing = {1.0: -0.1, 0.8: -0.05, 0.6: 0.5, 0.4: 0.6}
    assert baseline.certified_weight(weights, counts, 300, losing, 0.02, 0.05) is None
