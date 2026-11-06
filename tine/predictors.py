"""Per-model success predictors: X (n, d) embeddings -> P (n, M) success probabilities.

All predictors learn from outcomes (did model j succeed on similar tasks), never
from which model a previous router chose. NaN labels (pending) are ignored.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class KNNSuccessPredictor:
    """Retrieval memory: success rate of each model on the k most similar past tasks.

    Smoothed toward each model's overall rate so small neighbourhoods don't give 0 or 1.
    """

    def __init__(self, k=20, prior_strength=2.0):
        self.k = k
        self.prior_strength = prior_strength

    def fit(self, X, Y):
        self._Y = np.asarray(Y, dtype=float)
        self._prior = np.nanmean(self._Y, axis=0)
        self._index = NearestNeighbors(n_neighbors=min(self.k, len(X)), metric="cosine",
                                       algorithm="brute").fit(X)
        return self

    def predict_proba(self, X):
        _, neighbors = self._index.kneighbors(X)
        outcomes = self._Y[neighbors]  # (n, k, M)
        known = ~np.isnan(outcomes)
        successes = np.where(known, outcomes, 0.0).sum(axis=1)
        a = self.prior_strength
        return (successes + a * self._prior) / (known.sum(axis=1) + a)


class _PerModelClassifier:
    """One binary classifier per model."""

    def _make(self):
        raise NotImplementedError

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        self._models = []
        for j in range(Y.shape[1]):
            known = ~np.isnan(Y[:, j])
            labels = Y[known, j]
            if len(np.unique(labels)) < 2:
                self._models.append(float(labels.mean()) if len(labels) else 0.0)
            else:
                self._models.append(self._make().fit(X[known], labels.astype(int)))
        return self

    def predict_proba(self, X):
        columns = [
            np.full(len(X), m) if isinstance(m, float) else m.predict_proba(X)[:, 1]
            for m in self._models
        ]
        return np.column_stack(columns)


class LogisticSuccessPredictor(_PerModelClassifier):
    def __init__(self, C=1.0):
        self.C = C

    def _make(self):
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))


class BoostedTreeSuccessPredictor(_PerModelClassifier):
    """Gradient-boosted trees (scikit-learn; stand-in for LightGBM, not in the lockfile)."""

    def __init__(self, max_iter=200, learning_rate=0.05, seed=0):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.seed = seed

    def _make(self):
        return HistGradientBoostingClassifier(max_iter=self.max_iter,
                                              learning_rate=self.learning_rate,
                                              random_state=self.seed)


PREDICTORS = {
    "knn": KNNSuccessPredictor,
    "logreg": LogisticSuccessPredictor,
    "gbt": BoostedTreeSuccessPredictor,
}


class IsotonicCalibrated:
    """Wrap a predictor and map its raw scores to calibrated probabilities.

    The isotonic maps are fit on out-of-fold training predictions, so the
    calibration set stays untouched for certifying tau.
    """

    def __init__(self, make_base, n_folds=5, seed=0):
        self.make_base = make_base
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        oof = np.zeros_like(Y)
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for fit_idx, held_idx in folds.split(X):
            oof[held_idx] = self.make_base().fit(X[fit_idx], Y[fit_idx]).predict_proba(X[held_idx])
        self._maps = []
        for j in range(Y.shape[1]):
            known = ~np.isnan(Y[:, j])
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._maps.append(iso.fit(oof[known, j], Y[known, j]))
        self._base = self.make_base().fit(X, Y)
        return self

    def predict_proba(self, X):
        raw = self._base.predict_proba(X)
        return np.column_stack([m.predict(raw[:, j]) for j, m in enumerate(self._maps)])


def make_predictor(name="knn", calibrate=True, seed=0, **kwargs):
    if name not in PREDICTORS:
        raise ValueError(f"Unknown predictor {name!r}; choose from {sorted(PREDICTORS)}")
    def make_base():
        return PREDICTORS[name](**kwargs)
    return IsotonicCalibrated(make_base, seed=seed) if calibrate else make_base()
