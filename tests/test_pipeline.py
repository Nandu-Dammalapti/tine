import json

import numpy as np

from tine.__main__ import main
from tine.baselines import AvengersProRouter, KNNRouter, oracle_choices
from tine.data import RoutingTable, make_synthetic_results
from tine.evaluate import routing_metrics
from tine.pipeline import run_experiment
from tine.predictors import make_predictor


def test_oracle_is_cheapest_success():
    success = np.array([[0, 1, 1], [0, 0, 0]])
    cost = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert oracle_choices(success, cost).tolist() == [1, 0]


def test_predictors_return_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    Y = (rng.random((120, 3)) < 0.5).astype(float)
    Y[0, 0] = np.nan  # pending outcome is skipped
    for name in ("knn", "logreg", "gbt"):
        P = make_predictor(name).fit(X, Y).predict_proba(X[:7])
        assert P.shape == (7, 3) and np.all((P >= 0) & (P <= 1))


def test_baseline_routers_choose_valid_models():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    success = (rng.random((200, 3)) < 0.7).astype(float)
    cost = np.tile([1.0, 2.0, 5.0], (200, 1))
    for router, param in ((KNNRouter(), 0.5), (AvengersProRouter(n_clusters=5), 0.5)):
        prepared = router.fit(X, success, cost).prepare(X[:10])
        choices = router.choose(prepared, param)
        assert choices.shape == (10,) and choices.min() >= 0 and choices.max() <= 2


def test_metrics_against_frontier():
    success = np.array([[1, 1], [0, 1]])
    cost = np.array([[1.0, 4.0], [1.0, 4.0]])
    m = routing_metrics(np.array([0, 0]), success, cost, frontier=1)
    assert m["savings_pct"] == 75.0
    assert m["quality_loss"] == 0.5 and m["regret_rate"] == 0.5


def test_end_to_end_on_synthetic(tmp_path):
    table = RoutingTable.from_results(make_synthetic_results(n_tasks=1500, seed=3))
    summary = run_experiment(table, epsilon=0.05, delta=0.1, min_samples=100, out_dir=tmp_path)
    assert summary["frontier"] == "frontier"
    assert set(summary["thresholds"]) == {"*"}
    assert (tmp_path / "cost_quality.png").stat().st_size > 1000
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["tasks"]["calibration"] == 300
    assert summary["router_latency_ms"]["p50"] < 50


def test_precomputed_embeddings_are_used_and_aligned(tmp_path):
    table = RoutingTable.from_results(make_synthetic_results(n_tasks=600, seed=4))
    embeddings = np.random.default_rng(0).normal(size=(len(table), 8))
    summary = run_experiment(table, embeddings=embeddings, embedding_source="random test",
                             min_samples=100, epsilon=0.05, delta=0.1)
    assert summary["embedding_source"] == "random test"
    assert summary["router_latency_ms"]["includes_embedding"] is False


def test_cli_demo(tmp_path, capsys):
    assert main(["demo", "--n-tasks", "800", "--min-samples", "100", "--out", str(tmp_path)]) == 0
    assert "Gacrux (certified)" in capsys.readouterr().out
