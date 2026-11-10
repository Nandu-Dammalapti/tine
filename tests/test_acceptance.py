import numpy as np
import pandas as pd
import pytest

from tine.__main__ import main
from tine.acceptance import ResponseCheck
from tine.cascade import accept, certify_policies, run_cascade_experiment
from tine.data import synthetic_fixture, write_outcomes


def token_data(n=600, seed=0):
    """Two models. Model 0 is wrong exactly when its response is long; model 1's
    response length says nothing."""
    rng = np.random.default_rng(seed)
    output_tokens = rng.integers(50, 4000, size=(n, 2)).astype(float)
    outcomes = np.column_stack([output_tokens[:, 0] < 2000, rng.random(n) < 0.7])
    return outcomes, np.full((n, 2), 300.0), output_tokens


def test_response_check_learns_per_model_signal_on_held_out_tasks():
    outcomes, input_tokens, output_tokens = token_data()
    train, calibration, held = slice(0, 300), slice(300, 400), slice(400, 600)
    check = ResponseCheck().fit((outcomes[train], input_tokens[train], output_tokens[train]),
                                (outcomes[calibration], input_tokens[calibration], output_tokens[calibration]))
    scores = check.predict_scores(input_tokens[held], output_tokens[held])
    assert scores.shape == (200, 2) and ((scores >= 0) & (scores <= 1)).all()
    informed = scores[:, 0] >= 0.9
    assert informed.any() and outcomes[held][informed, 0].mean() > 0.95
    assert abs(np.median(scores[:, 1]) - 0.7) < 0.1  # no signal: about the base rate


def test_response_check_handles_one_class_and_rejects_bad_input():
    outcomes, input_tokens, output_tokens = token_data(100)
    outcomes[:, 1] = True
    data = (outcomes, input_tokens, output_tokens)
    scores = ResponseCheck().fit(data, data).predict_scores(input_tokens, output_tokens)
    assert np.allclose(scores[:, 1], 1.0)
    with pytest.raises(ValueError, match="Fit"):
        ResponseCheck().predict_scores(input_tokens, output_tokens)
    for bad in [(outcomes.astype(float), input_tokens, output_tokens),
                (outcomes, np.full_like(input_tokens, np.nan), output_tokens),
                (outcomes, input_tokens, -output_tokens), (outcomes, input_tokens[:, :1], output_tokens)]:
        with pytest.raises(ValueError):
            ResponseCheck().fit(bad, bad)
    with pytest.raises(ValueError):
        ResponseCheck().fit(data, data).predict_scores(input_tokens[:, :1], output_tokens[:, :1])


def test_accept_combines_a_check_with_a_score_threshold():
    passes = np.array([[True, True], [True, False]])
    scores = np.array([[0.9, 0.2], [0.4, 0.9]])
    assert accept(passes, scores, None) is passes
    assert accept(passes, scores, 0.5).tolist() == [[True, False], [False, False]]


def test_certified_search_picks_the_variant_that_holds_on_the_certify_part():
    n = 400
    outcomes = np.ones((n, 2), dtype=bool)
    outcomes[: n // 4, 0] = False
    costs = np.tile([1.0, 10.0], (n, 1))
    everything = np.ones((n, 2), dtype=bool)  # a check that catches nothing
    weak = (outcomes, costs, everything, None)
    strong = (outcomes, costs, outcomes, None)
    assert certify_policies([[0]], [("weak", weak, weak)], 1, epsilon=0.02, delta=0.05) == ([], None, None)
    chain, label, bound = certify_policies([[0]], [("weak", weak, weak), ("strong", strong, strong)], 1,
                                           epsilon=0.02, delta=0.05)
    assert chain == [0] and label == "strong" and bound <= 0.02
    with pytest.raises(ValueError):
        certify_policies([[0]], [], 1, epsilon=0.02, delta=0.05)


def token_frame(n_tasks=500, seed=5):
    """Synthetic outcomes where each cheaper model's wrong answers are long."""
    frame = synthetic_fixture(n_tasks=n_tasks, seed=seed)
    rng = np.random.default_rng(seed)
    wrong = ~frame.success.astype(bool) & (frame.model != "frontier")
    frame["input_tokens"] = 300
    frame["output_tokens"] = np.where(wrong, rng.integers(3000, 6000, len(frame)),
                                      rng.integers(100, 1500, len(frame)))
    return frame


def test_response_check_experiment_certifies_without_any_other_check(tmp_path):
    frame = token_frame()
    result = run_cascade_experiment(frame, "frontier", tmp_path, seeds=range(2), epsilons=(0.02,),
                                    catches=(0.0,), response_check=True)
    runs = pd.read_csv(tmp_path / "cascade-runs.csv")
    assert set(runs.method) == {"uncertified", "certified", "response_certified"}
    plain = runs[runs.method == "certified"]
    response = runs[runs.method == "response_certified"]
    assert (plain.chain == "frontier").all()          # a check that catches nothing certifies nothing
    assert (response.chain != "frontier").all() and response.response_tau.notna().all()
    assert (response.savings > 0).all() and (response.degradation_rate <= 0.02).all()
    assert result["response_check"]["features"] == ["input_tokens", "output_tokens"]
    aucs = result["response_check"]["held_out_auc_mean"]
    assert "frontier" not in aucs and all(value > 0.9 for value in aucs.values())
    assert result["fractions"] == [0.2, 0.1, 0.15, 0.4, 0.15]
    assert "response_certified" in (tmp_path / "report.md").read_text()


def test_response_check_requires_known_token_counts_and_valid_settings(tmp_path):
    frame = token_frame(200)
    missing = frame.copy()
    missing["output_tokens"] = missing["output_tokens"].astype(float)
    missing.loc[missing.index[0], "output_tokens"] = np.nan
    with pytest.raises(ValueError, match="input_tokens and output_tokens"):
        run_cascade_experiment(missing, "frontier", tmp_path, response_check=True)
    with pytest.raises(ValueError):
        run_cascade_experiment(frame, "frontier", tmp_path, response_check=True, response_thresholds=())
    with pytest.raises(ValueError):
        run_cascade_experiment(frame, "frontier", tmp_path, response_check="yes")
    with pytest.raises(ValueError, match="5 positive shares"):
        run_cascade_experiment(frame, "frontier", tmp_path, response_check=True, fit_fractions=(0.5, 0.5))


def test_response_check_cli(tmp_path, capsys):
    write_outcomes(token_frame(300, seed=6), tmp_path / "o.jsonl")
    main(["cascade", "--input", str(tmp_path / "o.jsonl"), "--frontier", "frontier",
          "--output", str(tmp_path / "out"), "--seeds", "1", "--epsilon", "0.02", "--catch", "0",
          "--response-check", "--response-threshold", "0.8", "0.9",
          "--fit-fractions", "0.2", "0.1", "0.2", "0.3", "0.2"])
    assert "response_certified" in capsys.readouterr().out
