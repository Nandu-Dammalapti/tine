"""Command line: `python -m tine demo` or `python -m tine run ...`. CPU only."""

import argparse
import json
import sys

from tine.data import RoutingTable, load_llmrouterbench, make_synthetic_results, read_results
from tine.features import load_embeddings
from tine.pipeline import run_experiment


def _common(parser):
    parser.add_argument("--predictor", default="knn", choices=["knn", "logreg", "gbt"])
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--per-task-type", action="store_true",
                        help="certify one tau per task type instead of one overall")
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="output folder (default under artifacts/)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tine")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run on invented data (no downloads)")
    demo.add_argument("--n-tasks", type=int, default=6000)
    _common(demo)

    run = sub.add_parser("run", help="run on graded results")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", help="CSV/JSONL with the tine.schema columns")
    source.add_argument("--llmrouterbench", help="folder holding <dataset>/<split>/<model>/*.json")
    run.add_argument("--datasets", nargs="*", help="LLMRouterBench datasets to keep")
    run.add_argument("--models", nargs="*", help="models to keep")
    run.add_argument("--tenant", default=None)
    run.add_argument("--frontier", default=None, help="frontier model name (default: most accurate)")
    run.add_argument("--embeddings", default=None, help=".npz from colab/01_embed_prompts.ipynb")
    _common(run)

    args = parser.parse_args(argv)
    if args.command == "demo":
        frame = make_synthetic_results(n_tasks=args.n_tasks, seed=args.seed)
        note = "SYNTHETIC data invented by tine.data.make_synthetic_results; not evidence"
        frontier, embeddings, source_name = None, None, None
        out = args.out or "artifacts/demo"
    else:
        if args.results:
            frame = read_results(args.results)
            note = f"results file {args.results}"
        else:
            frame = load_llmrouterbench(args.llmrouterbench, datasets=args.datasets,
                                        models=args.models)
            note = f"LLMRouterBench at {args.llmrouterbench}"
        if args.models and args.results:
            frame = frame[frame["model"].isin(args.models)]
        frontier, out = args.frontier, args.out or "artifacts/run"
        embeddings = source_name = None
    table = RoutingTable.from_results(frame, tenant=getattr(args, "tenant", None))
    if args.command == "run" and args.embeddings:
        embeddings = load_embeddings(args.embeddings, table.task_ids)
        source_name = f"precomputed {args.embeddings}"

    summary = run_experiment(
        table, frontier=frontier, predictor=args.predictor, epsilon=args.epsilon,
        delta=args.delta, per_task_type=args.per_task_type, embeddings=embeddings,
        embedding_source=source_name, min_samples=args.min_samples, seed=args.seed,
        out_dir=out, data_note=note)
    print_summary(summary)
    return 0


def print_summary(s):
    g = s["gacrux_test"]
    print(f"Data: {s['data_note']}")
    print(f"Embeddings: {s['embedding_source']}")
    print(f"Tasks: {s['tasks']}")
    print(f"Frontier: {s['frontier']}   epsilon={s['epsilon']}  delta={s['delta']}")
    for group, t in s["thresholds"].items():
        print(f"  tau[{group}] = {t['tau']}  (n={t['n']}; {t['reason']})")
    print(f"Gacrux (certified) on test: savings {g['savings_pct']:.1f}%  "
          f"quality loss {100 * g['quality_loss']:.2f} pts  regret {100 * g['regret_rate']:.2f}%  "
          f"oracle gap closed {g['oracle_gap_closed']:.2f}")
    print("  traffic: " + json.dumps({m: round(v, 3) for m, v in g["traffic_share"].items()}))
    print("Best savings within epsilon, picked on test (flatters baselines):")
    for name, p in s["best_savings_within_epsilon_test_selected"].items():
        print(f"  {name}: " + ("none" if p is None else f"{p['savings_pct']:.1f}%"))
    oracle = s["baselines_test"]["Oracle (hindsight ceiling)"]
    print(f"Oracle ceiling: {oracle['savings_pct']:.1f}% savings")
    lat = s["router_latency_ms"]
    print(f"Router latency per request: p50 {lat['p50']:.2f} ms, p99 {lat['p99']:.2f} ms "
          f"(embedding included: {lat['includes_embedding']})")
    if "outputs" in s:
        print(f"Wrote summary.json, calibration_steps.csv, cost_quality.png to {s['outputs']}")


if __name__ == "__main__":
    sys.exit(main())
