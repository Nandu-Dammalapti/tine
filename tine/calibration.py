"""Certify tau on held-out calibration data.

Loss per task: 1 if the frontier model succeeded and the routed model failed, else 0.
Its mean (the "regret rate") is at least the accuracy drop versus always-frontier,
so bounding it by epsilon bounds the quality loss by epsilon.

Procedure (Learn-then-Test with fixed-sequence testing, Angelopoulos et al. 2021):
for tau on a fixed grid from cautious (high) to aggressive (low), test
H0: regret(tau) > epsilon with the exact binomial p-value P(Binom(n, epsilon) <= k).
Reject while p <= delta; stop at the first tau that fails. Every rejected tau is
then safe simultaneously with probability >= 1 - delta, without a per-threshold
correction, and we keep the lowest. p <= delta is the same as the one-sided
Clopper-Pearson upper bound at level 1 - delta being <= epsilon.

Assumptions: calibration tasks and future tasks are exchangeable (i.i.d.) draws
from the same distribution, and outcomes are known for both the frontier and the
routed model. Drift, a new model, or a new task mix breaks this; recalibrate.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta, binom

from tine.routing import FRONTIER_ONLY, route_with_threshold

DEFAULT_GRID = np.round(np.linspace(1.0, 0.0, 101), 2)


def regret(choices, success, frontier):
    """Per-task loss: frontier succeeded and the chosen model failed."""
    chosen = success[np.arange(len(choices)), choices]
    return ((success[:, frontier] == 1) & (chosen == 0)).astype(int)


def clopper_pearson_upper(k, n, delta):
    """One-sided (1 - delta) upper confidence bound on a binomial rate."""
    return 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))


def min_calibration_samples(epsilon, delta):
    """Fewest calibration tasks that could certify any tau (needs zero observed regrets)."""
    return math.ceil(math.log(delta) / math.log(1 - epsilon))


@dataclass
class ThresholdResult:
    tau: float | None
    epsilon: float
    delta: float
    n: int
    reason: str
    steps: list = field(default_factory=list)


def select_threshold(P, success, est_cost, frontier, epsilon, delta, grid=DEFAULT_GRID):
    """Lowest tau certified by fixed-sequence testing; FRONTIER_ONLY if none."""
    if not 0 < epsilon < 1 or not 0 < delta < 1:
        raise ValueError("epsilon and delta must be in (0, 1)")
    grid = np.asarray(grid, dtype=float)
    if np.any(np.diff(grid) >= 0):
        raise ValueError("grid must be strictly decreasing (cautious to aggressive)")
    if np.isnan(success).any():
        raise ValueError("Calibration needs known outcomes for every model")
    n = len(P)
    certified, steps = FRONTIER_ONLY, []
    for tau in grid:
        choices = route_with_threshold(P, est_cost, frontier, float(tau))
        k = int(regret(choices, success, frontier).sum())
        p_value = float(binom.cdf(k, n, epsilon)) if n else 1.0
        passed = p_value <= delta
        steps.append({
            "tau": float(tau),
            "n": n,
            "regrets": k,
            "empirical_regret": k / n if n else float("nan"),
            "upper_bound": clopper_pearson_upper(k, n, delta) if n else 1.0,
            "p_value": p_value,
            "certified": passed,
            "share_routed_cheaper": float(np.mean(choices != frontier)) if n else 0.0,
        })
        if not passed:
            break
        certified = float(tau)
    reason = ("certified" if certified is not FRONTIER_ONLY
              else "no threshold certified; all traffic goes to the frontier model")
    return ThresholdResult(certified, epsilon, delta, n, reason, steps)


def select_thresholds_by_group(P, success, est_cost, groups, frontier, epsilon, delta,
                               min_samples=200, grid=DEFAULT_GRID):
    """Certify one tau per group (e.g. task type). Each group's statement holds
    separately with probability >= 1 - delta. Small groups get FRONTIER_ONLY."""
    groups = np.asarray(groups)
    floor = max(min_samples, min_calibration_samples(epsilon, delta))
    results = {}
    for group in np.unique(groups):
        rows = groups == group
        if rows.sum() < floor:
            results[str(group)] = ThresholdResult(
                FRONTIER_ONLY, epsilon, delta, int(rows.sum()),
                f"only {int(rows.sum())} calibration tasks (< {floor}); frontier only")
            continue
        results[str(group)] = select_threshold(P[rows], success[rows], est_cost[rows],
                                               frontier, epsilon, delta, grid)
    return results
