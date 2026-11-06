"""The routing rule: cheapest model with P(success) >= tau, else the frontier model."""

from dataclasses import dataclass, field

import numpy as np

# "No threshold was certified": every request goes to the frontier model.
# Deliberately not tau = 1.0, which would still let a model with a score of
# exactly 1.0 through.
FRONTIER_ONLY = None


class CostModel:
    """Expected cost of each model per task type, from training data.

    The router can't see a task's real cost before calling a model, so it uses the
    mean logged cost (retries and output length included) for that task type.
    """

    def fit(self, task_types, cost):
        task_types = np.asarray(task_types)
        self._global = np.nanmean(cost, axis=0)
        self._by_type = {t: np.nanmean(cost[task_types == t], axis=0)
                         for t in np.unique(task_types)}
        return self

    def estimate(self, task_types) -> np.ndarray:
        return np.vstack([self._by_type.get(t, self._global) for t in task_types])


def route_with_threshold(P, est_cost, frontier, tau):
    """Return the chosen model index for each row of P.

    A model qualifies when it is estimated cheaper than the frontier model and its
    success probability is at least tau. The cheapest qualifying model wins; if none
    qualifies, or tau is FRONTIER_ONLY, the frontier model is used.
    """
    P = np.asarray(P, dtype=float)
    choices = np.full(len(P), frontier, dtype=int)
    if tau is FRONTIER_ONLY:
        return choices
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1] or FRONTIER_ONLY, got {tau}")
    cheaper = est_cost < est_cost[:, [frontier]]
    qualifies = cheaper & (P >= tau)
    masked = np.where(qualifies, est_cost, np.inf)
    has_choice = np.isfinite(masked).any(axis=1)
    choices[has_choice] = masked[has_choice].argmin(axis=1)
    return choices


@dataclass
class ThresholdRouter:
    """A certified router: one tau per task type (or one for all, key "*")."""

    models: list
    frontier: int
    taus: dict = field(default_factory=dict)

    def tau_for(self, group):
        if group in self.taus:
            return self.taus[group]
        return self.taus.get("*", FRONTIER_ONLY)  # unseen task type -> frontier

    def route(self, P, est_cost, groups):
        groups = np.asarray(groups)
        choices = np.full(len(P), self.frontier, dtype=int)
        for group in np.unique(groups):
            rows = groups == group
            choices[rows] = route_with_threshold(P[rows], est_cost[rows], self.frontier,
                                                 self.tau_for(group))
        return choices
