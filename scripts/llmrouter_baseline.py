#!/usr/bin/env python3
"""Run routers from the LLMRouter library on existing graded outcomes.

The library (https://github.com/ulab-uiuc/LLMRouter, MIT) is not vendored: pass a
checkout with --llmrouter-dir. Its router and trainer classes for kNN, SVM, MLP,
matrix factorisation and Elo run unmodified, with the hyperparameters of its own
benchmark pipeline (``benchmark_pipeline/configs/hparams.yaml``). As that pipeline
does, this script trains each router on recorded outcomes and reads the choice from
the fitted model directly, so the library's text encoder is never loaded and no
model API is called. Query vectors come from an NPZ in scripts/embed.py format.
Read the third-party code before running it.

These routers have no quality guarantee and no cost setting of their own. The
library's benchmark makes them cost-aware by training on
``alpha * performance - beta * cost`` (both min-max normalised over the training
rows); this script does the same with ``alpha = weight`` and ``beta = 1 - weight``.
At weight 1 the label is "a model that succeeded", ties going to the first row of a
query: rows are written frontier first, then by falling mean cost, so weight 1 is
the cautious end and lower weights move to cheaper models.

Each seed splits tasks into train / calibrate / test. Per router it reports every
weight on the test part, and two picks made on the calibrate part and scored on the
test part: ``uncertified`` (most savings with degradation <= epsilon there) and
``certified`` (weights tested from cautious to aggressive with a Clopper-Pearson
bound, stopping at the first failure; Tine's certification applied to their router).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.calibration import clopper_pearson_upper, evaluate  # noqa: E402
from tine.data import read_outcomes  # noqa: E402
from tine.pipeline import complete_tasks, embedding_fingerprint, load_embeddings, matrix  # noqa: E402

DEFAULT_WEIGHTS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2)
ROUTERS = ("knnrouter", "svmrouter", "mlprouter", "mfrouter", "elorouter")
FRACTIONS = (0.5, 0.25, 0.25)


def load_library(llmrouter_dir: Path):
    llmrouter_dir = Path(llmrouter_dir).resolve()
    if not (llmrouter_dir / "llmrouter" / "models" / "knnrouter").is_dir():
        raise FileNotFoundError(f"No llmrouter/models/knnrouter under {llmrouter_dir}")
    sys.path.insert(0, str(llmrouter_dir))
    import yaml
    from llmrouter.models.elorouter import EloRouter, EloRouterTrainer
    from llmrouter.models.knnrouter import KNNRouter, KNNRouterTrainer
    from llmrouter.models.mfrouter import MFRouter, MFRouterTrainer
    from llmrouter.models.mlprouter import MLPRouter, MLPTrainer
    from llmrouter.models.svmrouter import SVMRouter, SVMRouterTrainer

    hparams = yaml.safe_load((llmrouter_dir / "benchmark_pipeline" / "configs" / "hparams.yaml").read_text())
    classes = {"knnrouter": (KNNRouter, KNNRouterTrainer), "svmrouter": (SVMRouter, SVMRouterTrainer),
               "mlprouter": (MLPRouter, MLPTrainer), "mfrouter": (MFRouter, MFRouterTrainer),
               "elorouter": (EloRouter, EloRouterTrainer)}
    return classes, hparams


def reward(success: np.ndarray, cost: np.ndarray, weight: float) -> np.ndarray:
    """The library's training target: alpha * norm(performance) - beta * norm(cost),
    min-max normalised over all training rows. At weight 1 it is the raw outcome, as
    in the library (which skips the transform for alpha 1, beta 0)."""
    performance = success.astype(float)
    if weight == 1.0:
        return performance
    span = lambda values: (values - values.min()) / (values.max() - values.min() + 1e-12)  # noqa: E731
    return weight * span(performance) - (1 - weight) * span(cost)


def training_rows(task_ids, outcomes, costs, models, order, weight: float) -> list[dict]:
    """Routing rows in the library's format. ``order`` is the model order within a
    query; the library breaks ties in favour of the first row."""
    values = reward(outcomes, costs, weight)
    rows = []
    for position, task_id in enumerate(task_ids):
        for column in order:
            rows.append({"task_name": "tine", "query": task_id, "model_name": models[column],
                         "performance": float(values[position, column]), "embedding_id": position,
                         "input_tokens": 0, "output_tokens": 0})
    return rows


def fit_and_route(name, classes, hparams, scratch: Path, rows, train_vectors, route_vectors, seed: int):
    """Train one library router and return the model name it picks per routed query."""
    import torch
    import yaml

    torch.manual_seed(seed)
    np.random.seed(seed)
    scratch.mkdir(parents=True, exist_ok=True)
    routing = scratch / "routing_train.jsonl"
    with routing.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    torch.save(torch.from_numpy(np.ascontiguousarray(train_vectors, dtype=np.float32)), scratch / "train.pt")
    params = dict(hparams.get(name) or {})
    if name == "mfrouter":
        params["text_dim"] = int(train_vectors.shape[1])
    saved = scratch / "model.pkl"
    config = scratch / "config.yaml"
    config.write_text(yaml.safe_dump({
        "data_path": {"routing_data_train": str(routing), "query_embedding_data": str(scratch / "train.pt")},
        "model_path": {"ini_model_path": "", "save_model_path": str(saved), "load_model_path": str(saved)},
        "metric": {"weights": {"performance": 1, "cost": 0, "llm_judge": 0}}, "hparam": params}))
    router_class, trainer_class = classes[name]
    frame = pd.DataFrame(rows)
    labels = set(frame.loc[frame.groupby("query")["performance"].idxmax(), "model_name"])
    if name in ("knnrouter", "svmrouter", "mlprouter") and len(labels) == 1:
        # Every training label is one model; a classifier cannot be fitted on one class.
        return np.array([next(iter(labels))] * len(route_vectors))
    router = router_class(str(config))
    trainer = trainer_class(router, device="cpu")
    trainer.train()
    if name == "knnrouter":
        return np.asarray(router.knn_model.predict(route_vectors))
    if name == "svmrouter":
        return np.asarray(router.svm_model.predict(route_vectors))
    points = torch.from_numpy(np.ascontiguousarray(route_vectors, dtype=np.float32))
    if name == "mlprouter":
        return np.array([router.idx_to_model[int(index)] for index in trainer.model.predict(points).tolist()])
    if name == "mfrouter":
        trainer.model.eval()
        with torch.no_grad():
            picks = [int(torch.argmax(trainer.model.score_all(trainer.model.project_text(point))))
                     for point in points]
        return np.array([router.idx_to_model[index] for index in picks])
    from llmrouter.utils import load_model

    return np.array([load_model(str(saved)).index[0]] * len(route_vectors))  # Elo: one model for all


def certified_weight(weights, degradations, counts, savings, epsilon: float, delta: float):
    """Fixed-sequence test from the cautious weight down: every weight before the
    first whose bound exceeds epsilon is valid; the one with most savings is used."""
    valid = []
    for weight in weights:
        if clopper_pearson_upper(degradations[weight], counts, delta) > epsilon:
            break
        valid.append(weight)
    best = max(valid, key=lambda weight: savings[weight]) if valid else None
    return best if best is not None and savings[best] > 0 else None  # never worse than the frontier alone


def run_baseline(frame, frontier: str, embeddings, llmrouter_dir, output, *, routers=ROUTERS,
                 weights=DEFAULT_WEIGHTS, seeds=range(10), epsilon=(0.04, 0.02), delta: float = 0.05,
                 library_commit: str | None = None) -> dict:
    weights = sorted({float(weight) for weight in weights}, reverse=True)
    epsilons = [float(value) for value in epsilon]
    if not weights or any(not 0 <= weight <= 1 for weight in weights):
        raise ValueError("Require performance weights between zero and one")
    frame, tasks, excluded = complete_tasks(frame, frontier)
    if tasks.tenant.nunique() != 1:
        raise ValueError("Run the baseline on one tenant at a time")
    lookup = load_embeddings(embeddings, tasks)
    vectors = np.stack([np.asarray(lookup[(task.tenant, task.task_id)], dtype=np.float32)
                        for task in tasks.itertuples()])
    models = sorted(frame.model.unique())
    frontier_index = models.index(frontier)
    outcomes = matrix(frame, tasks, models, "success", boolean=True)
    costs = matrix(frame, tasks, models, "cost_usd")
    mean_cost = costs.mean(axis=0)
    order = [frontier_index] + sorted((column for column in range(len(models)) if column != frontier_index),
                                      key=lambda column: -mean_cost[column])
    classes, hparams = load_library(llmrouter_dir)
    task_ids = tasks.task_id.to_numpy()
    runs, picks = [], []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # The library prints progress for every fit; keep it in a log next to the results.
    with tempfile.TemporaryDirectory() as scratch, \
            (output / "llmrouter-library.log").open("w", encoding="utf-8") as library_log:
        for seed in seeds:
            shuffled = np.random.default_rng(seed).permutation(len(tasks))
            first, second = int(FRACTIONS[0] * len(tasks)), int((FRACTIONS[0] + FRACTIONS[1]) * len(tasks))
            train, calibrate, test = shuffled[:first], shuffled[first:second], shuffled[second:]
            held = np.concatenate([calibrate, test])
            for name in routers:
                on_calibrate, on_test, choices = {}, {}, {}
                for weight in weights:
                    rows = training_rows(task_ids[train], outcomes[train], costs[train], models, order, weight)
                    print(f"== seed {seed} {name} weight {weight}", file=library_log, flush=True)
                    with contextlib.redirect_stdout(library_log):
                        names = fit_and_route(name, classes, hparams, Path(scratch) / f"{seed}-{name}-{weight}",
                                              rows, vectors[train], vectors[held], seed)
                    chosen = np.array([models.index(model) for model in names])
                    choices[weight] = chosen
                    on_calibrate[weight] = evaluate(chosen[:len(calibrate)], outcomes[calibrate],
                                                    costs[calibrate], frontier_index)
                    on_test[weight] = evaluate(chosen[len(calibrate):], outcomes[test], costs[test], frontier_index)
                    metrics = on_test[weight]
                    runs.append({"router": name, "seed": seed, "performance_weight": weight,
                                 "test_tasks": len(test), "savings": metrics["savings_fraction"],
                                 "accuracy": metrics["accuracy"], "frontier_accuracy": metrics["frontier_accuracy"],
                                 "degradation_rate": metrics["degradation_rate"],
                                 "frontier_share": metrics["frontier_count"] / len(test)})
                savings = {weight: on_calibrate[weight]["savings_fraction"] for weight in weights}
                counts = {weight: int(round(on_calibrate[weight]["degradation_rate"] * len(calibrate)))
                          for weight in weights}
                for limit in epsilons:
                    allowed = [weight for weight in weights
                               if on_calibrate[weight]["degradation_rate"] <= limit and savings[weight] > 0]
                    chosen_weights = {
                        "uncertified": max(allowed, key=lambda weight: savings[weight]) if allowed else None,
                        "certified": certified_weight(weights, counts, len(calibrate), savings, limit, delta)}
                    for method, weight in chosen_weights.items():
                        policy = (choices[weight][len(calibrate):] if weight is not None
                                  else np.full(len(test), frontier_index))
                        metrics = evaluate(policy, outcomes[test], costs[test], frontier_index)
                        picks.append({"router": name, "method": method, "epsilon": limit, "seed": seed,
                                      "picked_weight": weight, "test_tasks": len(test),
                                      "savings": metrics["savings_fraction"], "accuracy": metrics["accuracy"],
                                      "frontier_accuracy": metrics["frontier_accuracy"],
                                      "degradation_rate": metrics["degradation_rate"],
                                      "over_epsilon": bool(metrics["degradation_rate"] > limit)})
    runs, picks = pd.DataFrame(runs), pd.DataFrame(picks)
    curve = (runs.groupby(["router", "performance_weight"])
             .agg(savings_mean=("savings", "mean"), accuracy_mean=("accuracy", "mean"),
                  frontier_accuracy=("frontier_accuracy", "mean"), degradation_mean=("degradation_rate", "mean"),
                  degradation_max=("degradation_rate", "max"), frontier_share=("frontier_share", "mean"))
             .reset_index().sort_values(["router", "performance_weight"], ascending=[True, False]))
    summary = (picks.groupby(["router", "method", "epsilon"])
               .agg(savings_mean=("savings", "mean"), accuracy_mean=("accuracy", "mean"),
                    frontier_accuracy=("frontier_accuracy", "mean"), degradation_max=("degradation_rate", "max"),
                    over_epsilon_runs=("over_epsilon", "sum"),
                    frontier_only_runs=("picked_weight", lambda column: int(column.isna().sum())))
               .reset_index())
    result = {
        "frontier": frontier, "models": models, "complete_tasks": len(tasks), "excluded_tasks": len(excluded),
        "seeds": len(list(seeds)), "epsilons": epsilons, "delta": delta, "fractions": FRACTIONS,
        "weights": weights, "routers": list(routers),
        "library": {"repository": "https://github.com/ulab-uiuc/LLMRouter", "commit": library_commit,
                    "hyperparameters": "benchmark_pipeline/configs/hparams.yaml (mfrouter text_dim set to the "
                                       "embedding size)",
                    "replaced": "Nothing in the routers or trainers. Choices are read from the fitted models, "
                                "as the library's benchmark pipeline does, so its text encoder is not used."},
        "embedding_sha256": embedding_fingerprint(lookup, tasks),
        "scope": "These routers give no bound on quality loss. 'certified' applies Tine's fixed-sequence "
                 "Clopper-Pearson test to their cost weight; 'uncertified' picks the weight by its observed "
                 "degradation on the calibrate part.",
        "summary": summary.to_dict(orient="records"),
    }
    runs.to_csv(output / "llmrouter-runs.csv", index=False)
    picks.to_csv(output / "llmrouter-picks.csv", index=False)
    curve.to_csv(output / "llmrouter-curve.csv", index=False)
    summary.to_csv(output / "llmrouter-summary.csv", index=False)
    (output / "llmrouter-summary.json").write_text(
        json.dumps(result, indent=2, default=float, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llmrouter-dir", type=Path, required=True, help="Checkout of the library")
    parser.add_argument("--llmrouter-commit", help="Commit of that checkout, recorded in the summary")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routers", nargs="+", default=list(ROUTERS), choices=ROUTERS)
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.04, 0.02])
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args(argv)
    result = run_baseline(read_outcomes(args.input), args.frontier, args.embeddings, args.llmrouter_dir,
                          args.output, routers=args.routers, weights=args.weights, seeds=range(args.seeds),
                          epsilon=args.epsilon, delta=args.delta, library_commit=args.llmrouter_commit)
    print(pd.DataFrame(result["summary"]).round(4).to_string(index=False))
    print(f"Summary: {args.output / 'llmrouter-summary.json'}")


if __name__ == "__main__":
    main()
