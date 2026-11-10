import math

import numpy as np
import pytest

from tine.calibration import calibrate, clopper_pearson_upper, evaluate, minimum_risk_tasks


def example(n=1000):
    scores = np.tile([0.7, 0.9], (n, 1))
    costs = np.tile([0.1, 1.0], (n, 1))
    outcomes = np.ones((n, 2), dtype=bool)
    return scores, costs, outcomes


def test_exact_zero_event_bound_and_extremes():
    assert clopper_pearson_upper(0, 100, 0.05) == pytest.approx(1 - 0.05 ** 0.01)
    assert clopper_pearson_upper(100, 100, 0.05) == 1
    assert clopper_pearson_upper(0, 0, 0.05) == 1


def test_lowest_threshold_and_multiple_comparisons():
    scores, costs, outcomes = example()
    result = calibrate(scores, costs, outcomes, 1, [0.8, 0.1, 0.5], family_size=4)
    assert result["tau"] == 0.1
    assert result["force_frontier"] is False
    assert result["corrected_alpha"] == pytest.approx(0.05 / 4)  # family only, not grid size
    assert result["upper_bound"] == pytest.approx(clopper_pearson_upper(0, 1000, 0.05 / 4))
    assert [c["tau"] for c in result["candidates"]] == [0.8, 0.5, 0.1]  # cautious first
    assert result["thresholds_tested"] == 3


def test_small_calibration_forces_frontier_even_at_score_one():
    scores, costs, outcomes = example(10)
    scores[:, 0] = 1.0
    result = calibrate(scores, costs, outcomes, 1, [1.0], epsilon=0.01)
    assert result["force_frontier"] is True
    assert result["tau"] is None
    assert result["upper_bound"] == 0  # the selected policy is explicitly frontier
    assert result["candidates"][0]["upper_bound"] > 0.01
    assert result["candidates"][0]["nonfrontier_count"] == 10


def test_no_cheap_routes_on_calibration_still_needs_binomial_bound():
    scores, costs, outcomes = example(10)
    result = calibrate(scores, costs, outcomes, 1, [0.99], epsilon=0.01)
    assert result["force_frontier"]
    assert result["candidates"][0]["nonfrontier_count"] == 0
    assert result["candidates"][0]["upper_bound"] > 0


def test_minimum_sample_for_zero_events():
    epsilon, delta, hypotheses = 0.01, 0.05, 5
    needed = math.ceil(math.log(delta / hypotheses) / math.log(1 - epsilon))
    assert minimum_risk_tasks(epsilon, delta, hypotheses) == needed
    assert minimum_risk_tasks(0, delta, hypotheses) is None
    assert minimum_risk_tasks(1, delta, hypotheses) == 1
    for n, expected_fallback in [(needed - 1, True), (needed, False)]:
        scores, costs, outcomes = example(n)
        result = calibrate(scores, costs, outcomes, 1, [0.5], epsilon, delta, hypotheses)
        assert result["force_frontier"] is expected_fallback


def test_degradation_is_not_net_accuracy_loss():
    scores, costs, outcomes = example(1000)
    outcomes[:200, 1] = False  # cheap model improves these tasks
    outcomes[200:250, 0] = False  # but degrades fifty frontier successes
    result = calibrate(scores, costs, outcomes, 1, [0.5], epsilon=0.02)
    assert result["force_frontier"]
    assert result["candidates"][0]["degradations"] == 50
    metrics = evaluate(np.zeros(1000, dtype=int), outcomes, costs, 1)
    assert metrics["quality_loss"] == pytest.approx(-0.15)
    assert metrics["degradation_rate"] == 0.05


def test_empty_calibration_and_grid_are_explicit_fallbacks():
    scores, costs, outcomes = example(0)
    assert calibrate(scores, costs, outcomes, 1, [0.5])["reason"] == "empty_calibration"
    scores, costs, outcomes = example()
    result = calibrate(scores, costs, outcomes, 1, [])
    assert result["force_frontier"]
    assert result["reason"] == "empty_threshold_grid"
    assert result["corrected_alpha"] is None


@pytest.mark.parametrize("kwargs", [
    {"epsilon": float("nan")}, {"epsilon": -0.1}, {"delta": 0},
    {"delta": 1}, {"family_size": 0}, {"family_size": True},
    {"frontier_index": True}, {"thresholds": [float("nan")]},
])
def test_bad_calibration_parameters_rejected(kwargs):
    scores, costs, outcomes = example(10)
    parameters = {"frontier_index": 1, "thresholds": [0.5], **kwargs}
    with pytest.raises(ValueError):
        calibrate(scores, costs, outcomes, **parameters)


def test_unknown_outcomes_and_nonfinite_scores_or_costs_rejected():
    scores, costs, outcomes = example(10)
    with pytest.raises(ValueError, match="boolean"):
        calibrate(scores, costs, outcomes.astype(float), 1, [0.5])
    scores[0, 0] = np.nan
    with pytest.raises(ValueError, match="scores"):
        calibrate(scores, costs, outcomes, 1, [0.5])
    scores[0, 0] = 0.5
    costs[0, 0] = -1
    with pytest.raises(ValueError, match="costs"):
        calibrate(scores, costs, outcomes, 1, [0.5])


@pytest.mark.parametrize("field", ["scores", "estimated_costs"])
@pytest.mark.parametrize("value", [0.9 + 4j, False])
def test_calibration_rejects_coercible_nonreal_measurements(field, value):
    scores, costs, outcomes = example(200)
    values = {"scores": scores, "estimated_costs": costs}
    values[field] = values[field].astype(object)
    values[field][0, 0] = value
    with pytest.raises(ValueError, match="scores|costs"):
        calibrate(**values, outcomes=outcomes, frontier_index=1, thresholds=[0.5])


@pytest.mark.parametrize("costs", [np.array([[1 + 4j, 2]]), [[False, 1.0]]])
def test_evaluation_rejects_costs_that_would_change_during_conversion(costs):
    with pytest.raises(ValueError, match="costs"):
        evaluate([0], np.ones((1, 2), dtype=bool), costs, 1)


def test_metrics_cost_per_success_oracle_and_zero_denominators():
    outcomes = np.array([[True, True], [False, True], [False, False]])
    costs = np.array([[1, 4], [2, 5], [1, 6]], dtype=float)
    metrics = evaluate(np.array([0, 1, 1]), outcomes, costs, 1)
    assert metrics["total_cost_usd"] == 12
    assert metrics["cost_per_completed_task_usd"] == 6
    assert metrics["oracle_total_cost_usd"] == 12
    assert metrics["savings_fraction"] == pytest.approx(0.2)
    assert metrics["oracle_gap_closed_fraction"] == 1
    assert metrics["quality_loss"] == 0
    zero = evaluate(np.array([0]), np.array([[False, False]]), np.zeros((1, 2)), 1)
    assert zero["savings_fraction"] is None
    assert zero["cost_per_completed_task_usd"] is None
    assert zero["oracle_gap_closed_fraction"] is None


def test_invalid_choice_indices_rejected():
    _, costs, outcomes = example(3)
    for choices in [[0.0, 0.0, 0.0], [0, -1, 0], [0, 2, 0], [0, 0]]:
        with pytest.raises(ValueError, match="choice_indices"):
            evaluate(choices, outcomes, costs, 1)


def test_fixed_sequence_stops_at_first_failure():
    # Three models: 0 cheapest (safe), 1 mid (degrades 20%), 2 frontier.
    n = 1000
    scores = np.tile([0.6, 0.9, 0.0], (n, 1))
    costs = np.tile([0.1, 0.5, 1.0], (n, 1))
    outcomes = np.ones((n, 3), dtype=bool)
    outcomes[: n // 5, 1] = False
    result = calibrate(scores, costs, outcomes, 2, [1.0, 0.9, 0.5])
    # 1.0 passes (all frontier); 0.9 routes to the risky mid model and fails;
    # 0.5 would pass (cheapest model is safe) but is never tested.
    assert result["tau"] == 1.0
    assert [c["tested"] for c in result["candidates"]] == [True, True, False]
    assert [c["certified"] for c in result["candidates"]] == [True, False, False]
    assert result["candidates"][2]["upper_bound"] <= 0.02  # diagnostic only
    assert result["thresholds_tested"] == 2


def test_fixed_sequence_failure_rate_stays_below_delta():
    """Across many calibration draws, the chosen threshold's true degradation rate
    exceeds epsilon at most about delta of the time, with a 50-threshold grid and
    no division of delta across thresholds."""
    epsilon, delta, n, fail_rate = 0.05, 0.1, 300, 0.12
    grid = np.round(np.linspace(1.0, 0.02, 50), 2)
    rng = np.random.default_rng(0)
    costs = np.tile([0.1, 1.0], (n, 1))
    violations, trials = 0, 400
    for _ in range(trials):
        score = rng.random(n)
        scores = np.column_stack([score, np.zeros(n)])
        outcomes = np.ones((n, 2), dtype=bool)
        outcomes[:, 0] = rng.random(n) >= fail_rate
        result = calibrate(scores, costs, outcomes, 1, grid, epsilon, delta)
        # The cheap model is used when score >= tau; its true degradation rate:
        if result["tau"] is not None and fail_rate * (1 - result["tau"]) > epsilon:
            violations += 1
    assert violations / trials <= delta + 0.03
