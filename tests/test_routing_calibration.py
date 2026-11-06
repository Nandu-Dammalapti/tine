import numpy as np
import pytest

from tine.calibration import (min_calibration_samples, regret, select_threshold,
                              select_thresholds_by_group)
from tine.routing import FRONTIER_ONLY, ThresholdRouter, route_with_threshold

# Models: 0 cheap, 1 mid, 2 frontier, 3 pricier than frontier.
COSTS = np.array([[1.0, 2.0, 5.0, 9.0]])


def test_frontier_only_ignores_perfect_scores():
    P = np.ones((3, 4))
    choices = route_with_threshold(P, np.repeat(COSTS, 3, 0), frontier=2, tau=FRONTIER_ONLY)
    assert (choices == 2).all()


def test_cheapest_qualifying_model_and_never_pricier_than_frontier():
    P = np.array([[0.5, 0.95, 0.99, 1.0],
                  [0.96, 0.97, 0.99, 1.0],
                  [0.1, 0.2, 0.3, 1.0]])
    choices = route_with_threshold(P, np.repeat(COSTS, 3, 0), frontier=2, tau=0.9)
    assert choices.tolist() == [1, 0, 2]


def test_rejects_bad_tau():
    with pytest.raises(ValueError):
        route_with_threshold(np.ones((1, 4)), COSTS, frontier=2, tau=1.5)


def test_router_sends_unseen_groups_to_frontier():
    router = ThresholdRouter(["a", "b", "c", "d"], frontier=2, taus={"coding": 0.5})
    choices = router.route(np.ones((2, 4)), np.repeat(COSTS, 2, 0), np.array(["coding", "new"]))
    assert choices.tolist() == [0, 2]


def _calibration_case(n, cheap_fail_rate, seed=0):
    rng = np.random.default_rng(seed)
    success = np.ones((n, 4))
    success[:, 0] = rng.random(n) >= cheap_fail_rate
    P = np.tile([0.97, 0.0, 0.0, 0.0], (n, 1))
    return P, success, np.repeat(COSTS, n, 0)


def test_certifies_when_cheap_model_is_safe():
    P, success, cost = _calibration_case(1000, cheap_fail_rate=0.0)
    result = select_threshold(P, success, cost, 2, epsilon=0.02, delta=0.05)
    assert result.tau == 0.0  # nothing ever fails, every tau passes


def test_refuses_when_cheap_model_is_unsafe():
    P, success, cost = _calibration_case(1000, cheap_fail_rate=0.10)
    result = select_threshold(P, success, cost, 2, epsilon=0.02, delta=0.05)
    assert result.tau == 0.98  # only taus above 0.97 (all-frontier) pass
    assert result.steps[-1]["certified"] is False


def test_fixed_sequence_stops_at_first_failure():
    # At tau 0.9 a risky model is used; at 0.5 a safe one would be, but testing has stopped.
    n = 1000
    success = np.ones((n, 4))
    success[: n // 5, 1] = 0  # mid model fails 20%
    P = np.tile([0.6, 0.9, 0.0, 0.0], (n, 1))
    result = select_threshold(P, success, np.repeat(COSTS, n, 0), 2, 0.02, 0.05,
                              grid=[1.0, 0.9, 0.5])
    assert result.tau == 1.0
    assert len(result.steps) == 2


def test_small_groups_fall_back_to_frontier():
    P, success, cost = _calibration_case(100, cheap_fail_rate=0.0)
    results = select_thresholds_by_group(P, success, cost, np.array(["a"] * 100), 2, 0.02, 0.05)
    assert results["a"].tau is FRONTIER_ONLY


def test_min_calibration_samples():
    assert min_calibration_samples(0.02, 0.05) == 149


def test_guarantee_holds_in_simulation():
    """Over many calibration draws, the certified tau's true regret exceeds epsilon
    at most about delta of the time."""
    epsilon, delta, n = 0.05, 0.1, 300
    rng = np.random.default_rng(0)
    # 50 candidate taus; tau_i routes to the cheap model with true regret rising as tau drops.
    grid = np.round(np.linspace(1.0, 0.02, 50), 2)
    def true_regret(tau):
        return 0.12 * (1 - tau)  # cheap model used when score >= tau, fails 12%
    violations = 0
    for trial in range(400):
        s = rng.random(n)
        fails = rng.random(n) < 0.12
        success = np.ones((n, 4))
        success[:, 0] = ~fails
        P = np.column_stack([s, np.zeros(n), np.zeros(n), np.zeros(n)])
        tau = select_threshold(P, success, np.repeat(COSTS, n, 0), 2, epsilon, delta, grid).tau
        if tau is not None and true_regret(tau) > epsilon:
            violations += 1
    assert violations / 400 <= delta + 0.03


def test_regret_counts_only_frontier_right_routed_wrong():
    success = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
    assert regret(np.array([0, 0, 0]), success, frontier=2).tolist() == [1, 0, 1]
