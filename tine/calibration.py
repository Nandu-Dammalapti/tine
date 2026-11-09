"""Held-out routing calibration with a simultaneous conservative risk bound.

The measured Bernoulli event is ``frontier succeeded AND chosen model failed``.
Its probability upper-bounds net accuracy loss against the frontier.

Thresholds are tested by fixed-sequence testing (Learn-then-Test, Angelopoulos
et al. 2021): from the most cautious threshold to the most aggressive, each is
tested at the same level and testing stops at the first failure. Because the
order is fixed before seeing data, this controls the chance of certifying any
unsafe threshold at that level without dividing it across thresholds. A
Bonferroni correction still covers the caller-declared family of
tenant/task-type/method policies. Calibration requires
independent, identically distributed held-out tasks from the deployment group;
it is not a guarantee under distribution shift or dependent/reused tasks.
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil, log, log1p
from numbers import Integral, Real

import numpy as np
from scipy.stats import beta

from tine.routing import route_indices


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _probability(value: object, name: str, *, open_interval: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability")
    result = float(value)
    valid = 0 < result < 1 if open_interval else 0 <= result <= 1
    if not np.isfinite(result) or not valid:
        raise ValueError(f"{name} must be {'strictly between' if open_interval else 'between'} 0 and 1")
    return result


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """Exact one-sided binomial upper bound with confidence ``1 - alpha``.

    No observations provide no evidence, so the bound is 1.0. The caller must
    account for threshold selection and any other simultaneous comparisons.
    """
    n = _integer(n, "n")
    k = _integer(k, "k")
    alpha = _probability(alpha, "alpha", open_interval=True)
    if k > n:
        raise ValueError("k cannot exceed n")
    if n == 0 or k == n:
        return 1.0
    # isf avoids cancellation in 1-alpha for very small corrected alpha.
    return float(beta.isf(alpha, k + 1, n - k))


def minimum_risk_tasks(epsilon: float, delta: float, comparisons: int) -> int | None:
    """Smallest nonempty calibration set that can pass with zero degradations.

    ``comparisons`` is the Bonferroni family (groups x methods); fixed-sequence
    testing means the threshold grid size does not enter.
    """
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta", open_interval=True)
    comparisons = _integer(comparisons, "comparisons", minimum=1)
    alpha = delta / comparisons
    if alpha == 0:
        raise ValueError("corrected alpha underflows; reduce the hypothesis family")
    if epsilon == 0:
        return None
    if epsilon == 1:
        return 1
    # For zero degradations the bound is 1 - (delta / comparisons)**(1/n).
    return ceil(Fraction(log(alpha)) / Fraction(log1p(-epsilon)))


def _matrices(outcomes, costs, frontier_index):
    outcomes = np.asarray(outcomes)
    if outcomes.ndim != 2 or outcomes.shape[1] == 0:
        raise ValueError("outcomes must have shape (tasks, models) with at least one model")
    if outcomes.dtype != np.dtype(bool):
        raise ValueError("outcomes must contain known boolean success labels")
    costs = np.asarray(costs, dtype=float)
    if costs.shape != outcomes.shape or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("costs must match outcomes and be finite and nonnegative")
    frontier_index = _integer(frontier_index, "frontier_index")
    if frontier_index >= outcomes.shape[1]:
        raise ValueError("frontier_index is outside the model columns")
    return outcomes, costs, frontier_index


def calibrate(
    scores,
    estimated_costs,
    outcomes,
    frontier_index: int,
    thresholds,
    epsilon: float = 0.02,
    delta: float = 0.05,
    family_size: int = 1,
) -> dict:
    """Pick the lowest certified threshold, or force the frontier explicitly.

    ``scores`` and ``estimated_costs`` must be fixed independently of the risk
    calibration outcomes. ``thresholds`` must be a predeclared finite grid.
    ``family_size`` counts all groups and model-selection methods covered by the
    requested confidence statement, even if some yield no data. Each threshold is
    tested at ``delta / family_size``, from highest to lowest, stopping at the
    first one whose upper bound exceeds epsilon (fixed-sequence testing). Later
    thresholds are still evaluated for diagnostics but marked untested and can
    never be selected.

    The conservative degradation bound can certify an expected accuracy loss
    at most epsilon under the stated sampling assumptions. It does not certify
    every individual request or validate calibration under distribution shift.
    """
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta", open_interval=True)
    family_size = _integer(family_size, "family_size", minimum=1)
    outcomes, estimated_costs, frontier_index = _matrices(
        outcomes, estimated_costs, frontier_index
    )
    scores = np.asarray(scores, dtype=float)
    if (
        scores.shape != outcomes.shape
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("scores must match outcomes and contain finite probabilities")
    # Most cautious first: the order is fixed before any calibration outcome is seen.
    grid = sorted({_probability(value, "threshold") for value in thresholds}, reverse=True)
    alpha = delta / family_size if grid else None
    if alpha is not None and alpha == 0:
        raise ValueError("corrected alpha underflows; reduce the hypothesis family")
    n = outcomes.shape[0]
    result = {
        "tau": None,
        "force_frontier": True,
        "n": n,
        "degradations": 0,
        "upper_bound": 0.0,
        "epsilon": epsilon,
        "delta": delta,
        "thresholds_in_grid": len(grid),
        "thresholds_tested": 0,
        "family_size": family_size,
        "corrected_alpha": alpha,
        "bound": "one-sided Clopper-Pearson; fixed-sequence over thresholds (high to low); "
                 "Bonferroni across groups and methods",
        "loss": "frontier_success_and_selected_failure",
        "reason": "empty_calibration" if not n else "no_certified_threshold",
        "candidates": [],
    }
    if not n or not grid:
        if n and not grid:
            result["reason"] = "empty_threshold_grid"
        return result

    rows = np.arange(n)
    still_testing = True
    for tau in grid:
        selected = route_indices(scores, estimated_costs, tau, frontier_index)
        degradations = int(np.count_nonzero(outcomes[:, frontier_index] & ~outcomes[rows, selected]))
        upper = clopper_pearson_upper(degradations, n, alpha)
        passed = still_testing and upper <= epsilon
        result["candidates"].append(
            {
                "tau": tau,
                "degradations": degradations,
                "upper_bound": upper,
                "nonfrontier_count": int(np.count_nonzero(selected != frontier_index)),
                "tested": still_testing,
                "certified": passed,
            }
        )
        if still_testing:
            result["thresholds_tested"] += 1
        if passed:
            # Each pass moves the choice to a lower (more aggressive) threshold.
            result.update(
                tau=tau,
                force_frontier=False,
                degradations=degradations,
                upper_bound=upper,
                reason="certified_threshold",
            )
        else:
            still_testing = False  # first failure ends the sequence
    return result


def evaluate(choice_indices, outcomes, observed_costs, frontier_index: int) -> dict:
    """Evaluate a frozen policy on fully observed held-out task/model results.

    Observed costs must include the actual cost of each completed attempt or
    retry bundle, according to the dataset's definition. They are for evaluation,
    never inputs for choosing a deployable policy. The oracle is hindsight only:
    cheapest successful model, or frontier when every model failed.
    """
    outcomes, observed_costs, frontier_index = _matrices(
        outcomes, observed_costs, frontier_index
    )
    choices = np.asarray(choice_indices)
    n, model_count = outcomes.shape
    if choices.shape != (n,) or choices.dtype.kind not in "iu":
        raise ValueError("choice_indices must be one integer model index per task")
    if (choices < 0).any() or (choices >= model_count).any():
        raise ValueError("choice_indices contains an invalid model index")
    rows = np.arange(n)
    selected_success = outcomes[rows, choices]
    frontier_success = outcomes[:, frontier_index]
    cost = float(observed_costs[rows, choices].sum())
    frontier_cost = float(observed_costs[:, frontier_index].sum())
    completed = int(np.count_nonzero(selected_success))
    accuracy = float(selected_success.mean()) if n else None
    frontier_accuracy = float(frontier_success.mean()) if n else None
    degradation_count = int(np.count_nonzero(frontier_success & ~selected_success))
    oracle_costs = np.where(outcomes, observed_costs, np.inf)
    oracle_choices = np.argmin(oracle_costs, axis=1)
    oracle_choices[~outcomes.any(axis=1)] = frontier_index
    oracle_cost = float(observed_costs[rows, oracle_choices].sum())
    oracle_gap = frontier_cost - oracle_cost
    return {
        "n": n,
        "successes": completed,
        "accuracy": accuracy,
        "frontier_accuracy": frontier_accuracy,
        "quality_loss": frontier_accuracy - accuracy if n else None,
        "degradations": degradation_count,
        "degradation_rate": degradation_count / n if n else None,
        "total_cost_usd": cost,
        "mean_cost_usd": cost / n if n else None,
        "cost_per_completed_task_usd": cost / completed if completed else None,
        "frontier_total_cost_usd": frontier_cost,
        "cost_saved_usd": frontier_cost - cost,
        "savings_fraction": (frontier_cost - cost) / frontier_cost if frontier_cost else None,
        "oracle_total_cost_usd": oracle_cost,
        "oracle_gap_closed_fraction": (frontier_cost - cost) / oracle_gap if oracle_gap > 0 else None,
        "frontier_count": int(np.count_nonzero(choices == frontier_index)),
    }
