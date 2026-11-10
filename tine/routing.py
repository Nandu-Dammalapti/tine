"""CPU outcome-based routing predictors and deterministic cost-aware decisions.

TF-IDF is the default: no model weights or API calls are required. Optional fixed
embeddings must be supplied by the caller, which must verify their prompt hashes.
The cluster method is a simple outcome-cluster heuristic, not a reproduction of
Avengers-Pro. Probability calibration is distinct from risk calibration: these
scores alone never establish a quality guarantee.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from tine.numeric import real_array


TASK_COLUMNS = ["tenant", "task_id", "prompt", "task_type"]
METHODS = ("logistic", "knn", "cluster")


def _validate_tasks(frame: pd.DataFrame, *, unique: bool = True) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Tasks must be a DataFrame with unique column names")
    missing = set(TASK_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing task columns: {sorted(missing)}")
    tasks = frame[TASK_COLUMNS].copy()
    if tasks.empty:
        raise ValueError("Tasks must not be empty")
    for column in TASK_COLUMNS:
        if not tasks[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{column} must contain nonempty strings")
    if unique and tasks.duplicated(["tenant", "task_id"]).any():
        raise ValueError("Tasks must be unique by tenant/task_id")
    if tasks.groupby(["tenant", "task_id"])[["prompt", "task_type"]].nunique().gt(1).any().any():
        raise ValueError("A task must have one prompt and task_type")
    return tasks


def _validate_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    _validate_tasks(frame, unique=False)
    missing = {"model", "success", "cost_usd"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing outcome columns: {sorted(missing)}")
    result = frame.copy()
    if not result["model"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("model must contain nonempty strings")
    if result.duplicated(["tenant", "task_id", "model"]).any():
        raise ValueError("Duplicate tenant/task_id/model outcome")
    known = result["success"].notna()
    if not result.loc[known, "success"].map(
        lambda value: isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))
        and value in (0, 1)
    ).all():
        raise ValueError("Known success labels must be Boolean or zero/one")
    costs = real_array(result["cost_usd"], "cost_usd")
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("cost_usd must be finite and nonnegative")
    result["cost_usd"] = costs
    return result


def route_indices(
    scores: np.ndarray,
    costs: np.ndarray,
    tau: float,
    frontier_index: int,
    force_frontier: bool = False,
) -> np.ndarray:
    """Choose the cheapest eligible model, otherwise the frontier model.

    The frontier is always eligible, independently of its predicted score. A
    calibrated all-frontier policy must set ``force_frontier=True``; tau=1 alone
    does not exclude a cheaper model whose score is exactly one. Cost ties are
    resolved by column order. Predictions and costs must be finite and valid.
    """
    scores = real_array(scores, "scores")
    costs = real_array(costs, "costs")
    if scores.ndim != 2 or scores.shape != costs.shape or scores.shape[1] == 0:
        raise ValueError("scores and costs must be matching task-by-model matrices")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities")
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Costs must be finite and nonnegative")
    if isinstance(frontier_index, bool) or not isinstance(frontier_index, (int, np.integer)):
        raise ValueError("frontier_index must be an integer column index")
    if not 0 <= frontier_index < scores.shape[1]:
        raise ValueError("frontier_index is out of bounds")
    if isinstance(tau, (bool, np.bool_)) or not isinstance(tau, Real) or not 0 <= tau <= 1:
        raise ValueError("tau must be between zero and one")
    if not isinstance(force_frontier, (bool, np.bool_)):
        raise ValueError("force_frontier must be a boolean")
    if force_frontier:
        return np.full(scores.shape[0], frontier_index, dtype=int)
    eligible = scores >= tau
    eligible[:, frontier_index] = True
    return np.argmin(np.where(eligible, costs, np.inf), axis=1)


class Router:
    """One tenant's CPU predictors with separate held-out score calibration.

    Fit learns features, predictors, and estimated costs only from ``train``.
    ``score_calibration`` supplies outcome labels only for isotonic mappings.
    Both sets must be disjoint by task identity and prompt. Unknown outcomes are
    never converted to failure or success; every model must have some known
    training and score-calibration outcomes. The caller supplies a separate risk
    calibration set later to select a policy threshold.
    """

    def __init__(
        self,
        frontier: str,
        seed: int = 42,
        n_neighbors: int = 20,
        n_clusters: int = 8,
        embedding_lookup: dict[tuple[str, str], np.ndarray] | None = None,
    ) -> None:
        if not isinstance(frontier, str) or not frontier.strip():
            raise ValueError("frontier must be a nonempty model name")
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        self.frontier = frontier
        self.seed = seed
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.fitted_ = False
        self.embedding_lookup = None
        if embedding_lookup is not None:
            copied = {}
            dimension = None
            for key, value in embedding_lookup.items():
                if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(k, str) for k in key):
                    raise ValueError("Embedding keys must be (tenant, task_id) string tuples")
                vector = real_array(value, "Embedding vectors").copy()
                if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                    raise ValueError("Embedding vectors must be finite, nonempty one-dimensional arrays")
                if dimension is None:
                    dimension = vector.size
                if vector.size != dimension:
                    raise ValueError("Embedding vectors must have the same dimension")
                copied[key] = vector
            if not copied:
                raise ValueError("embedding_lookup must not be empty")
            self.embedding_lookup = copied

    def _check_tenant(self, tasks: pd.DataFrame) -> None:
        if tasks["tenant"].nunique() != 1 or tasks["tenant"].iloc[0] != self.tenant_:
            raise ValueError("Router is isolated to its fitted tenant")

    @staticmethod
    def _text(tasks: pd.DataFrame) -> list[str]:
        return (tasks["task_type"] + "\n" + tasks["prompt"]).tolist()

    def _features(self, tasks: pd.DataFrame, *, fit: bool = False) -> sparse.csr_matrix:
        if self.embedding_lookup is not None:
            keys = list(zip(tasks["tenant"], tasks["task_id"]))
            missing = [key for key in keys if key not in self.embedding_lookup]
            if missing:
                raise ValueError(f"Missing fixed embeddings for {len(missing)} tasks")
            text_features = normalize(sparse.csr_matrix(np.vstack([self.embedding_lookup[key] for key in keys])))
        else:
            if fit:
                self.vectorizer_ = TfidfVectorizer(
                    analyzer="char", ngram_range=(1, 3), max_features=4096, sublinear_tf=True,
                )
                text_features = self.vectorizer_.fit_transform(self._text(tasks))
            else:
                text_features = self.vectorizer_.transform(self._text(tasks))
        numeric = []
        for prompt in tasks["prompt"]:
            length = max(len(prompt), 1)
            numeric.append([
                min(np.log1p(length) / 10, 2),
                min(np.log1p(len(prompt.split())) / 10, 2),
                sum(character.isdigit() for character in prompt) / length,
                prompt.count("\n") / length,
            ])
        return normalize(sparse.hstack([text_features, sparse.csr_matrix(numeric)], format="csr"))

    def fit(self, train: pd.DataFrame, score_calibration: pd.DataFrame) -> Router:
        self.fitted_ = False
        train = _validate_outcomes(train)
        score_calibration = _validate_outcomes(score_calibration)
        train_tasks = train[TASK_COLUMNS].drop_duplicates(["tenant", "task_id"]).reset_index(drop=True)
        calibration_tasks = score_calibration[TASK_COLUMNS].drop_duplicates(["tenant", "task_id"]).reset_index(drop=True)
        if train["tenant"].nunique() != 1:
            raise ValueError("Fit each tenant separately")
        self.tenant_ = train["tenant"].iloc[0]
        self._check_tenant(calibration_tasks)
        if self.embedding_lookup is not None:
            self.embedding_lookup = {key: vector for key, vector in self.embedding_lookup.items() if key[0] == self.tenant_}
        train_keys = set(zip(train_tasks["tenant"], train_tasks["task_id"]))
        calibration_keys = set(zip(calibration_tasks["tenant"], calibration_tasks["task_id"]))
        if train_keys & calibration_keys or set(train_tasks["prompt"]) & set(calibration_tasks["prompt"]):
            raise ValueError("Training and score calibration must be disjoint by task and prompt")
        self.model_names = sorted(train["model"].unique().tolist())
        if self.frontier not in self.model_names:
            raise ValueError("Frontier model must have training outcomes")
        if set(score_calibration["model"]) - set(self.model_names):
            raise ValueError("Score calibration contains a model absent from training")
        self.frontier_index = self.model_names.index(self.frontier)
        self._model_costs = train.groupby("model")["cost_usd"].mean().to_dict()
        self._type_costs = train.groupby(["task_type", "model"])["cost_usd"].mean().to_dict()
        features = self._features(train_tasks, fit=True)
        unique_texts = len(set(self._text(train_tasks)))
        self.clusterer_ = MiniBatchKMeans(
            n_clusters=min(self.n_clusters, len(train_tasks), unique_texts),
            random_state=self.seed, n_init=3, max_iter=100, batch_size=min(1024, len(train_tasks)),
        )
        task_index = pd.MultiIndex.from_frame(train_tasks[["tenant", "task_id"]])
        self.predictors_ = {}
        with threadpool_limits(limits=1):
            clusters = self.clusterer_.fit_predict(features)
            for model in self.model_names:
                known = train[(train["model"] == model) & train["success"].notna()]
                if known.empty:
                    raise ValueError(f"Model {model!r} has no known training outcomes")
                indices = task_index.get_indexer(pd.MultiIndex.from_frame(known[["tenant", "task_id"]]))
                labels = known["success"].to_numpy(dtype=float)
                model_features = features[indices]
                if np.unique(labels).size == 1:
                    logistic = float(labels[0])
                else:
                    logistic = LogisticRegression(max_iter=500, solver="liblinear", random_state=self.seed)
                    logistic.fit(model_features, labels)
                neighbors = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(known)), metric="cosine", algorithm="brute")
                neighbors.fit(model_features)
                cluster_scores = np.full(self.clusterer_.n_clusters, labels.mean(), dtype=float)
                for cluster in range(self.clusterer_.n_clusters):
                    in_cluster = clusters[indices] == cluster
                    if in_cluster.any():
                        cluster_scores[cluster] = labels[in_cluster].mean()
                self.predictors_[model] = (logistic, neighbors, labels, cluster_scores)

            calibration_features = self._features(calibration_tasks)
            calibration_index = pd.MultiIndex.from_frame(calibration_tasks[["tenant", "task_id"]])
            self.calibrators_ = {method: {} for method in METHODS}
            for method in METHODS:
                raw = self._raw_scores(calibration_features, method)
                for column, model in enumerate(self.model_names):
                    known = score_calibration[(score_calibration["model"] == model) & score_calibration["success"].notna()]
                    if known.empty:
                        raise ValueError(f"Model {model!r} has no known score-calibration outcomes")
                    indices = calibration_index.get_indexer(pd.MultiIndex.from_frame(known[["tenant", "task_id"]]))
                    calibration = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                    calibration.fit(raw[indices, column], known["success"].to_numpy(dtype=float))
                    self.calibrators_[method][model] = calibration
        self.fitted_ = True
        return self

    def _raw_scores(self, features: sparse.csr_matrix, method: str) -> np.ndarray:
        scores = []
        cluster_ids = self.clusterer_.predict(features) if method == "cluster" else None
        for model in self.model_names:
            logistic, neighbors, labels, cluster_scores = self.predictors_[model]
            if method == "logistic":
                values = np.full(features.shape[0], logistic) if isinstance(logistic, float) else logistic.predict_proba(features)[:, 1]
            elif method == "knn":
                indices = neighbors.kneighbors(features, return_distance=False)
                values = labels[indices].mean(axis=1)
            elif method == "cluster":
                values = cluster_scores[cluster_ids]
            else:
                raise ValueError(f"Unknown method: {method!r}; choose one of {METHODS}")
            scores.append(values)
        return np.column_stack(scores)

    def _prediction_tasks(self, tasks: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "fitted_", False):
            raise ValueError("Fit the router before prediction")
        tasks = _validate_tasks(tasks)
        self._check_tenant(tasks)
        return tasks

    def predict_scores(self, tasks: pd.DataFrame, method: str = "logistic") -> np.ndarray:
        tasks = self._prediction_tasks(tasks)
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method!r}; choose one of {METHODS}")
        with threadpool_limits(limits=1):
            raw = self._raw_scores(self._features(tasks), method)
            return np.column_stack([
                self.calibrators_[method][model].predict(raw[:, column])
                for column, model in enumerate(self.model_names)
            ])

    def estimated_costs(self, tasks: pd.DataFrame) -> np.ndarray:
        tasks = self._prediction_tasks(tasks)
        return np.array([
            [self._type_costs.get((task_type, model), self._model_costs[model]) for model in self.model_names]
            for task_type in tasks["task_type"]
        ], dtype=float)
