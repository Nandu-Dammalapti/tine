"""Comparison routers. Each returns one model index per test task.

- best single: always the frontier model
- random: a random model with probability q, else frontier (q sweeps a curve)
- kNN: pick argmax(neighbour accuracy - lambda * neighbour cost)
- Avengers-Pro style: cluster prompts, score models per cluster on accuracy and cost
- oracle: cheapest model that succeeded, known only in hindsight (a ceiling, not a router)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def frontier_choices(n, frontier):
    return np.full(n, frontier, dtype=int)


def random_choices(n, n_models, frontier, q, seed=0):
    rng = np.random.default_rng(seed)
    choices = np.full(n, frontier, dtype=int)
    switch = rng.random(n) < q
    choices[switch] = rng.integers(n_models, size=int(switch.sum()))
    return choices


def oracle_choices(success, cost):
    """Cheapest model that succeeded; if none did, the cheapest model overall."""
    succeeded = np.where(success == 1, cost, np.inf)
    any_success = np.isfinite(succeeded).any(axis=1)
    return np.where(any_success, succeeded.argmin(axis=1), cost.argmin(axis=1))


class KNNRouter:
    """kNN baseline (cf. arXiv:2505.12601): neighbours' accuracy and cost per model."""

    def __init__(self, k=20):
        self.k = k

    def fit(self, X, success, cost):
        self._success, self._cost = success, cost
        self._scale = np.nanmean(cost, axis=0).max()
        self._index = NearestNeighbors(n_neighbors=min(self.k, len(X)), metric="cosine",
                                       algorithm="brute").fit(X)
        return self

    def prepare(self, X):
        _, neighbors = self._index.kneighbors(X)
        accuracy = np.nanmean(self._success[neighbors], axis=1)
        cost = np.nanmean(self._cost[neighbors], axis=1) / self._scale
        return accuracy, cost

    @staticmethod
    def choose(prepared, lam):
        accuracy, cost = prepared
        return (accuracy - lam * cost).argmax(axis=1)


class AvengersProRouter:
    """Re-implementation from the paper's description (Zhang et al., 2025): k-means on
    prompt embeddings; per cluster, min-max normalise each model's accuracy and cost;
    score = alpha * accuracy + (1 - alpha) * (1 - cost), summed over the top_p nearest
    clusters. Check against the authors' code before quoting it as a reproduction.
    """

    def __init__(self, n_clusters=32, top_p=3, seed=0):
        self.n_clusters = n_clusters
        self.top_p = top_p
        self.seed = seed

    def fit(self, X, success, cost):
        n_clusters = max(1, min(self.n_clusters, len(X) // 10))
        self._kmeans = KMeans(n_clusters=n_clusters, n_init=4, random_state=self.seed).fit(X)
        labels = self._kmeans.labels_
        n_models = success.shape[1]
        self._acc = np.zeros((n_clusters, n_models))
        self._cost = np.zeros((n_clusters, n_models))
        for c in range(n_clusters):
            rows = labels == c
            self._acc[c] = _minmax(np.nanmean(success[rows], axis=0))
            self._cost[c] = _minmax(np.nanmean(cost[rows], axis=0))
        return self

    def prepare(self, X):
        distances = self._kmeans.transform(X)
        top = np.argsort(distances, axis=1)[:, : min(self.top_p, distances.shape[1])]
        return self._acc[top].sum(axis=1), self._cost[top].sum(axis=1)

    @staticmethod
    def choose(prepared, alpha):
        # alpha * acc + (1 - alpha) * (1 - cost); the constant drops out of the argmax.
        accuracy, cost = prepared
        return (alpha * accuracy - (1 - alpha) * cost).argmax(axis=1)


def _minmax(values):
    low, high = np.nanmin(values), np.nanmax(values)
    return np.zeros_like(values) if high - low < 1e-12 else (values - low) / (high - low)
