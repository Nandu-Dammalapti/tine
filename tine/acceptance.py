"""Accept or escalate a cheaper model's answer from the size of its response.

Token counts are known as soon as a model has answered and are already in the
ledger, so this check needs no test run and no extra model call. For many models
a long response (long reasoning) goes with a wrong answer; the direction and
strength differ per model, so each model gets its own small classifier.

The score is a probability calibrated on held-out rows. It is evidence for a
policy to be certified on further held-out tasks, never a guarantee by itself.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tine.numeric import real_array

RESPONSE_COLUMNS = ("input_tokens", "output_tokens")


def _token_matrix(values, name: str) -> np.ndarray:
    values = real_array(values, name)
    if (values.ndim != 2 or not values.size or not np.isfinite(values).all()
            or (values < 0).any() or (values != np.floor(values)).any()):
        raise ValueError(f"{name} must be a nonempty task-by-model matrix of known nonnegative whole token counts")
    return values


def _features(input_tokens: np.ndarray, output_tokens: np.ndarray) -> np.ndarray:
    prompt, response = np.log1p(input_tokens), np.log1p(output_tokens)
    return np.column_stack([response, prompt, response ** 2, response * prompt])


class ResponseCheck:
    """Per-model probability that an answer is right, given prompt and response
    token counts. ``fit`` learns from one set of tasks and calibrates on another;
    both must be disjoint from every task the scores are later used on."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.fitted_ = False

    @staticmethod
    def _inputs(outcomes, input_tokens, output_tokens):
        outcomes = np.asarray(outcomes)
        if outcomes.ndim != 2 or outcomes.dtype != np.dtype(bool):
            raise ValueError("outcomes must be a task-by-model matrix of known Boolean labels")
        input_tokens = _token_matrix(input_tokens, "input_tokens")
        output_tokens = _token_matrix(output_tokens, "output_tokens")
        if not (outcomes.shape == input_tokens.shape == output_tokens.shape) or not len(outcomes):
            raise ValueError("outcomes and token counts must be matching nonempty matrices")
        return outcomes, input_tokens, output_tokens

    def fit(self, train, score_calibration) -> ResponseCheck:
        """``train`` and ``score_calibration`` are (outcomes, input_tokens, output_tokens)."""
        self.fitted_ = False
        train, calibration = self._inputs(*train), self._inputs(*score_calibration)
        if train[0].shape[1] != calibration[0].shape[1]:
            raise ValueError("Training and score calibration must cover the same models")
        self.models_ = []
        for column in range(train[0].shape[1]):
            labels = train[0][:, column]
            if labels.all() or not labels.any():
                classifier = float(labels[0])
            else:
                classifier = make_pipeline(StandardScaler(), LogisticRegression(
                    max_iter=1000, random_state=self.seed))
                classifier.fit(_features(train[1][:, column], train[2][:, column]), labels)
            calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            calibrator.fit(self._raw(classifier, calibration[1][:, column], calibration[2][:, column]),
                           calibration[0][:, column].astype(float))
            self.models_.append((classifier, calibrator))
        self.fitted_ = True
        return self

    @staticmethod
    def _raw(classifier, input_tokens, output_tokens) -> np.ndarray:
        if isinstance(classifier, float):
            return np.full(len(input_tokens), classifier)
        return classifier.predict_proba(_features(input_tokens, output_tokens))[:, 1]

    def predict_scores(self, input_tokens, output_tokens) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("Fit the response check before prediction")
        input_tokens = _token_matrix(input_tokens, "input_tokens")
        output_tokens = _token_matrix(output_tokens, "output_tokens")
        if input_tokens.shape != output_tokens.shape or input_tokens.shape[1] != len(self.models_):
            raise ValueError("Token counts must be matching matrices with one column per fitted model")
        return np.column_stack([
            calibrator.predict(self._raw(classifier, input_tokens[:, column], output_tokens[:, column]))
            for column, (classifier, calibrator) in enumerate(self.models_)])
