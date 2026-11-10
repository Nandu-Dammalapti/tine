#!/usr/bin/env python3
"""Run the official Avengers-Pro balance router on existing graded outcomes.

The official code (https://github.com/ZhangYiqun018/AvengersPro, MIT) is not
vendored: pass a checkout with --avengers-dir. Its clustering, per-cluster
scoring, routing and train/test split run unmodified. Only its embedding client
is replaced, by vectors from an NPZ in scripts/embed.py format, so no embedding
service is called. Read the third-party code before running it.

Avengers-Pro has no quality guarantee: its performance weight trades accuracy
for cost. Each seed therefore reports every weight on the official test split,
plus one weight picked on a random half of that split (most savings with
degradation <= epsilon there) and scored on the other half. No model is called.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.calibration import evaluate  # noqa: E402
from tine.data import read_outcomes  # noqa: E402
from tine.pipeline import complete_tasks, embedding_fingerprint, load_embeddings, matrix  # noqa: E402

DEFAULT_WEIGHTS = tuple(round(0.1 * step, 1) for step in range(11))


def official_rows(frame: pd.DataFrame, tasks: pd.DataFrame, models: list[str]) -> list[dict]:
    """One row per task in the official input format; ``index`` is the task row."""
    outcomes = matrix(frame, tasks, models, "success", boolean=True)
    costs = matrix(frame, tasks, models, "cost_usd")
    return [{
        "query": task.prompt, "dataset": task.task_type, "index": row,
        "records": {model: float(outcomes[row, column]) for column, model in enumerate(models)},
        "usages": {model: {"cost": float(costs[row, column])} for column, model in enumerate(models)},
    } for row, task in enumerate(tasks.itertuples())]


def load_official(avengers_dir: Path):
    """Import the official router with its embedding service and tokenizer replaced."""
    avengers_dir = Path(avengers_dir).resolve()
    if not (avengers_dir / "balance_cluster_router.py").is_file():
        raise FileNotFoundError(f"No balance_cluster_router.py under {avengers_dir}")
    sys.path.insert(0, str(avengers_dir))
    import balance_cluster_router
    import config as official_config
    import simple_cluster_router

    class NoEmbeddingService:
        def __init__(self, *args, **kwargs):
            pass

    class NoTokenizer:
        @staticmethod
        def encoding_for_model(name):
            return None

    simple_cluster_router.EmbeddingCache = NoEmbeddingService
    simple_cluster_router.tiktoken = NoTokenizer

    class SuppliedEmbeddingRouter(balance_cluster_router.BalanceClusterRouter):
        vectors: dict[str, np.ndarray] = {}

        def _generate_embeddings_concurrent(self, queries):
            return [self.vectors[query] for query in queries]

    return SuppliedEmbeddingRouter, official_config.SimpleClusterConfig


def summarize_picks(picks: pd.DataFrame) -> dict:
    return {"savings_mean": float(picks.savings.mean()), "accuracy_mean": float(picks.accuracy.mean()),
            "frontier_accuracy": float(picks.frontier_accuracy.mean()),
            "degradation_max": float(picks.degradation_rate.max()),
            "over_epsilon_runs": int(picks.over_epsilon.sum()),
            "frontier_only_runs": int(picks.picked_weight.isna().sum())}


def run_baseline(frame, frontier: str, embeddings, avengers_dir, output, *, weights=DEFAULT_WEIGHTS,
                 seeds=range(10), clusters: int = 60, top_k: int = 4, beta: float = 9.0,
                 epsilon: float = 0.02, avengers_commit: str | None = None) -> dict:
    weights = [float(weight) for weight in weights]
    seeds = [int(seed) for seed in seeds]
    if not weights or not seeds or any(not 0 <= weight <= 1 for weight in weights):
        raise ValueError("Require seeds and performance weights between zero and one")
    epsilons = [float(value) for value in (epsilon if isinstance(epsilon, (list, tuple)) else [epsilon])]
    if not epsilons or any(not 0 <= value <= 1 for value in epsilons):
        raise ValueError("epsilon must be between zero and one")
    epsilon = epsilons[0]  # the curve counts runs over the first limit
    frame, tasks, excluded = complete_tasks(frame, frontier)
    if tasks.tenant.nunique() != 1:
        raise ValueError("Run the baseline on one tenant at a time")
    lookup = load_embeddings(embeddings, tasks)
    models = sorted(frame.model.unique())
    frontier_index = models.index(frontier)
    outcomes = matrix(frame, tasks, models, "success", boolean=True)
    costs = matrix(frame, tasks, models, "cost_usd")
    router_class, config_class = load_official(avengers_dir)
    router_class.vectors = {task.prompt: np.asarray(lookup[(task.tenant, task.task_id)], dtype=float)
                            for task in tasks.itertuples()}
    logging.disable(logging.CRITICAL)  # the official code logs every step
    runs, picks = [], []
    with tempfile.TemporaryDirectory() as scratch:
        data_path = Path(scratch) / "official-input.jsonl"
        with data_path.open("w", encoding="utf-8") as stream:
            for row in official_rows(frame, tasks, models):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for seed in seeds:
            by_weight = {}
            for weight in weights:
                config = config_class(
                    data_path=str(data_path), n_clusters=clusters, train_ratio=0.7, seed=seed,
                    max_router=1, top_k=top_k, beta=beta, performance_weight=weight,
                    cost_sensitivity=1 - weight, embedding_api_key="unused",
                    embedding_model="supplied-embeddings", max_tokens=7500)
                router = router_class(config)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    train, test = router.load_and_split_data()
                    router.build_cluster_model(train)
                    details = router.evaluate_routing(test)["routing_details"]
                rows = np.array([detail["index"] for detail in details])
                choices = np.array([models.index(detail["selected_models"][0]) for detail in details])
                by_weight[weight] = (rows, choices)
                metrics = evaluate(choices, outcomes[rows], costs[rows], frontier_index)
                runs.append({"seed": seed, "performance_weight": weight, "test_tasks": len(rows),
                             "savings": metrics["savings_fraction"], "accuracy": metrics["accuracy"],
                             "frontier_accuracy": metrics["frontier_accuracy"],
                             "degradation_rate": metrics["degradation_rate"],
                             "frontier_share": metrics["frontier_count"] / len(rows)})
            # The official split does not depend on the weight, so halves line up.
            rows = by_weight[weights[0]][0]
            half = np.random.default_rng(seed).permutation(len(rows)) < len(rows) // 2
            on_half = {weight: evaluate(by_weight[weight][1][half], outcomes[rows[half]],
                                        costs[rows[half]], frontier_index) for weight in weights}
            for limit in epsilons:
                best, best_savings = None, 0.0
                for weight in weights:
                    metrics = on_half[weight]
                    if metrics["degradation_rate"] <= limit and metrics["savings_fraction"] > best_savings:
                        best, best_savings = weight, metrics["savings_fraction"]
                chosen = (by_weight[best][1] if best is not None
                          else np.full(len(rows), frontier_index))
                metrics = evaluate(chosen[~half], outcomes[rows[~half]], costs[rows[~half]], frontier_index)
                picks.append({"epsilon": limit, "seed": seed, "picked_weight": best,
                              "eval_tasks": int((~half).sum()),
                              "savings": metrics["savings_fraction"], "accuracy": metrics["accuracy"],
                              "frontier_accuracy": metrics["frontier_accuracy"],
                              "degradation_rate": metrics["degradation_rate"],
                              "over_epsilon": bool(metrics["degradation_rate"] > limit)})
    logging.disable(logging.NOTSET)
    runs, picks = pd.DataFrame(runs), pd.DataFrame(picks)
    curve = (runs.assign(over=runs.degradation_rate > epsilon).groupby("performance_weight")
             .agg(savings_mean=("savings", "mean"), accuracy_mean=("accuracy", "mean"),
                  frontier_accuracy=("frontier_accuracy", "mean"),
                  degradation_mean=("degradation_rate", "mean"),
                  degradation_max=("degradation_rate", "max"), over_epsilon_runs=("over", "sum"),
                  frontier_share=("frontier_share", "mean")).reset_index())
    result = {
        "frontier": frontier, "models": models, "complete_tasks": len(tasks),
        "excluded_tasks": len(excluded), "seeds": len(seeds), "epsilon": epsilon,
        "official_code": {"repository": "https://github.com/ZhangYiqun018/AvengersPro",
                          "commit": avengers_commit, "router": "BalanceClusterRouter",
                          "n_clusters": clusters, "top_k": top_k, "beta": beta, "train_ratio": 0.7,
                          "replaced": "Embedding client only; vectors supplied from an NPZ."},
        "embedding_sha256": embedding_fingerprint(lookup, tasks),
        "picked": summarize_picks(picks[picks.epsilon == epsilon]),
        "picked_by_epsilon": {str(limit): summarize_picks(picks[picks.epsilon == limit])
                              for limit in epsilons},
        "scope": "Uncertified: Avengers-Pro gives no bound on quality loss. The picked weight is chosen "
                 "on half of the official test split and scored on the other half.",
        "curve": curve.to_dict(orient="records"),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "avengers-runs.csv", index=False)
    picks.to_csv(output / "avengers-picks.csv", index=False)
    curve.to_csv(output / "avengers-curve.csv", index=False)
    (output / "avengers-summary.json").write_text(
        json.dumps(result, indent=2, default=float, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--avengers-dir", type=Path, required=True, help="Checkout of the official code")
    parser.add_argument("--avengers-commit", help="Commit of that checkout, recorded in the summary")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=60, help="Paper: 60")
    parser.add_argument("--top-k", type=int, default=4, help="Nearest clusters; paper: 4")
    parser.add_argument("--beta", type=float, default=9.0)
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.02],
                        help="quality limits for the held-out pick; the curve counts runs over the first")
    args = parser.parse_args(argv)
    result = run_baseline(read_outcomes(args.input), args.frontier, args.embeddings, args.avengers_dir,
                          args.output, weights=args.weights, seeds=range(args.seeds),
                          clusters=args.clusters, top_k=args.top_k, beta=args.beta,
                          epsilon=args.epsilon, avengers_commit=args.avengers_commit)
    print(json.dumps({"picked_by_epsilon": result["picked_by_epsilon"], "curve": result["curve"]},
                     indent=2, default=float))
    print(f"Summary: {args.output / 'avengers-summary.json'}")


if __name__ == "__main__":
    main()
