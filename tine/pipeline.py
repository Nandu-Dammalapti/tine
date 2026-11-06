"""End to end on CPU: split -> embed -> fit predictor -> certify tau -> evaluate vs baselines."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tine.baselines import (AvengersProRouter, KNNRouter, frontier_choices, oracle_choices,
                            random_choices)
from tine.calibration import DEFAULT_GRID, select_thresholds_by_group
from tine.data import RoutingTable, split_indices
from tine.evaluate import best_savings_within, plot_cost_quality, routing_metrics
from tine.features import TfidfEmbedder
from tine.predictors import make_predictor
from tine.routing import CostModel, ThresholdRouter, route_with_threshold


def pick_frontier(table: RoutingTable, train_idx) -> int:
    """Default frontier: most accurate model on training tasks (ties -> pricier one)."""
    accuracy = np.nanmean(table.success[train_idx], axis=0)
    cost = np.nanmean(table.cost[train_idx], axis=0)
    return int(np.lexsort((cost, accuracy))[-1])


def run_experiment(table: RoutingTable, *, frontier=None, predictor="knn", epsilon=0.02,
                   delta=0.05, per_task_type=False, embeddings=None, embedding_source=None,
                   min_samples=200, fractions=(0.6, 0.2, 0.2), seed=0, out_dir=None,
                   data_note=""):
    """Run the v1 evaluation for one tenant's table and return a summary dict.

    `embeddings` (n, d), aligned with `table.task_ids`, come from Colab; without them a
    TF-IDF stand-in is fit on the training prompts only.
    """
    n_before = len(table)
    if embeddings is not None and len(embeddings) != n_before:
        raise ValueError("embeddings must have one row per task in the table passed in")
    keep = table.complete_index()
    table = table.subset(keep)
    if embeddings is not None:
        embeddings = np.asarray(embeddings)[keep]
    if len(table) < 50:
        raise ValueError(f"Only {len(table)} tasks have outcomes for every model")
    train, cal, test = split_indices(table.task_types, fractions, seed)
    frontier = pick_frontier(table, train) if frontier is None else table.model_index(frontier)
    S, C, T = table.success, table.cost, table.task_types

    embedder = None
    if embeddings is None:
        embedder = TfidfEmbedder(seed=seed).fit(table.prompts[train])
        X = embedder.transform(table.prompts)
        embedding_source = "TF-IDF + SVD (CPU stand-in; use Colab embeddings for real runs)"
    else:
        X = np.asarray(embeddings, dtype=np.float32)

    model = make_predictor(predictor, seed=seed).fit(X[train], S[train])
    costs = CostModel().fit(T[train], C[train])
    groups = T if per_task_type else np.full(len(table), "*")

    # Certify tau on the calibration split only.
    P_cal = model.predict_proba(X[cal])
    thresholds = select_thresholds_by_group(P_cal, S[cal], costs.estimate(T[cal]), groups[cal],
                                            frontier, epsilon, delta, min_samples)
    router = ThresholdRouter(table.models, frontier, {g: r.tau for g, r in thresholds.items()})

    # Evaluate on the untouched test split.
    P_test, E_test = model.predict_proba(X[test]), costs.estimate(T[test])
    S_test, C_test = S[test], C[test]
    def metrics(choices):
        return routing_metrics(choices, S_test, C_test, frontier, table.models)
    gacrux = metrics(router.route(P_test, E_test, groups[test]))

    knn = KNNRouter().fit(X[train], S[train], C[train])
    knn_prep = knn.prepare(X[test])
    avengers = AvengersProRouter(seed=seed).fit(X[train], S[train], C[train])
    avengers_prep = avengers.prepare(X[test])
    curves = {
        "Gacrux tau sweep (uncertified)": [
            metrics(route_with_threshold(P_test, E_test, frontier, float(t)))
            for t in DEFAULT_GRID[::4]],
        "kNN router": [metrics(KNNRouter.choose(knn_prep, lam))
                       for lam in np.concatenate([[0.0], np.logspace(-3, 1, 20)])],
        "Avengers-Pro style": [metrics(AvengersProRouter.choose(avengers_prep, a))
                               for a in np.linspace(0, 1, 21)],
        "Random": [metrics(random_choices(len(test), len(table.models), frontier, q, seed))
                   for q in np.linspace(0, 1, 11)],
    }
    points = {
        f"Gacrux certified (eps={epsilon}, delta={delta})": gacrux,
        "Best single (frontier)": metrics(frontier_choices(len(test), frontier)),
        "Oracle (hindsight ceiling)": metrics(oracle_choices(S_test, C_test)),
    }

    summary = {
        "tenant": table.tenant,
        "data_note": data_note,
        "embedding_source": embedding_source,
        "predictor": f"{predictor} + isotonic calibration",
        "models": table.models,
        "frontier": table.models[frontier],
        "epsilon": epsilon,
        "delta": delta,
        "tasks": {"complete": len(table), "dropped_incomplete": n_before - len(table),
                  "train": len(train), "calibration": len(cal), "test": len(test)},
        "thresholds": {g: {"tau": r.tau, "n": r.n, "reason": r.reason}
                       for g, r in thresholds.items()},
        "gacrux_test": gacrux,
        "baselines_test": {name: p for name, p in points.items() if not name.startswith("Gacrux")},
        "best_savings_within_epsilon_test_selected": {
            name: best_savings_within(curve, epsilon) for name, curve in curves.items()},
        "router_latency_ms": _latency(embedder, X[test], table.prompts[test], T[test],
                                      groups[test], model, costs, router),
        "guarantee_note": (
            "Certified per group by Learn-then-Test fixed-sequence testing with exact "
            "binomial (Clopper-Pearson) p-values: with probability >= 1 - delta over the "
            "calibration draw, the expected regret rate (frontier right, routed model wrong) "
            "is <= epsilon, assuming calibration and future tasks are exchangeable. The test "
            "split is one draw; its regret can exceed epsilon by sampling noise."),
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        steps = [dict(group=g, **s) for g, r in thresholds.items() for s in r.steps]
        pd.DataFrame(steps).to_csv(out / "calibration_steps.csv", index=False)
        plot_cost_quality(curves, points, out / "cost_quality.png",
                          title=f"Cost vs quality on test split ({table.tenant})")
        summary["outputs"] = str(out)
    return summary


def _latency(embedder, X_test, prompts, task_types, groups, model, costs, router, n=200):
    """Per-request time for embed (TF-IDF only) + predict + route, one request at a time."""
    timings = []
    for i in range(min(n, len(prompts))):
        start = time.perf_counter()
        x = embedder.transform(prompts[i:i + 1]) if embedder is not None else X_test[i:i + 1]
        router.route(model.predict_proba(x), costs.estimate(task_types[i:i + 1]), groups[i:i + 1])
        timings.append(1000 * (time.perf_counter() - start))
    return {"p50": float(np.percentile(timings, 50)), "p99": float(np.percentile(timings, 99)),
            "includes_embedding": embedder is not None, "requests": len(timings)}
