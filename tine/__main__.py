"""Command-line entry points; no model calls are made by offline commands."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tine")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "demo"):
        run = commands.add_parser(name, help="Evaluate graded outcomes" if name == "run" else "Synthetic CPU end-to-end check")
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--epsilon", type=float, default=0.02)
        run.add_argument("--delta", type=float, default=0.05)
        run.add_argument("--seed", type=int, default=42)
        if name == "run":
            run.add_argument("--input", type=Path, required=True)
            run.add_argument("--frontier", required=True, help="Predeclared frontier baseline, not selected on test labels")
            run.add_argument("--embeddings", type=Path)
        else:
            run.add_argument("--tasks", type=int, default=2000)
    importer = commands.add_parser("import-llmrouterbench", help="Convert extracted upstream result JSON")
    importer.add_argument("--input", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    importer.add_argument("--tenant", default="benchmark")
    for option in ("dataset", "model", "split"):
        importer.add_argument(f"--{option}")
    importer.add_argument("--score-threshold", type=float)
    importer.add_argument("--run-selection", choices=("error", "latest"), default="error")
    combine = commands.add_parser("combine-outcomes", help="Pool outcome files into one calibration group")
    combine.add_argument("--input", type=Path, nargs="+", required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--task-type", help="Relabel every task to this one group, e.g. coding")
    combine.add_argument("--exclude-model", action="append", default=[])
    combine.add_argument("--keep-unpriced-models", action="store_true",
                         help="Keep models whose recorded cost is zero on every row")
    coding = commands.add_parser("import-coding-tasks", help="Prepare original coding prompts and checks")
    coding.add_argument("--dataset", required=True, choices=("humaneval", "mbpp"))
    coding.add_argument("--input", type=Path, required=True)
    coding.add_argument("--output", type=Path, required=True)
    coding.add_argument("--tenant", default="benchmark")
    preflight = commands.add_parser("preflight", help="Check outcome coverage and calibration sample sizes")
    preflight.add_argument("--input", type=Path, required=True)
    preflight.add_argument("--frontier", required=True)
    preflight.add_argument("--epsilon", type=float, default=0.02)
    preflight.add_argument("--delta", type=float, default=0.05)
    preflight.add_argument("--seed", type=int, default=42)
    inspect = commands.add_parser("inspect", help="Validate canonical outcomes and show models/coverage")
    inspect.add_argument("--input", type=Path, required=True)
    server = commands.add_parser("serve", help="Start the authenticated development proxy")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command in {"run", "demo"}:
        from tine.data import read_outcomes, synthetic_fixture
        from tine.pipeline import run_experiment

        if args.command == "demo":
            frame = synthetic_fixture(n_tasks=args.tasks, seed=args.seed)
            frontier = "frontier"
        else:
            frame, frontier = read_outcomes(args.input), args.frontier
        summary = run_experiment(frame, frontier, args.output, epsilon=args.epsilon,
                                 delta=args.delta, seed=args.seed,
                                 embeddings=getattr(args, "embeddings", None),
                                 synthetic=args.command == "demo")
        print(json.dumps(summary, indent=2, allow_nan=False))
        print(f"Report: {args.output / 'report.md'}")
    elif args.command == "import-llmrouterbench":
        from tine.data import import_llmrouterbench

        frame = import_llmrouterbench(args.input, args.output, tenant=args.tenant,
                                     dataset=args.dataset, model=args.model, split=args.split,
                                     score_threshold=args.score_threshold, run_selection=args.run_selection)
        print(json.dumps({"rows": len(frame), "tasks": len(frame[["tenant", "task_id"]].drop_duplicates()),
                          "models": sorted(frame.model.unique()), "output": str(args.output)}, indent=2))
    elif args.command == "combine-outcomes":
        from tine.data import combine_outcomes, read_outcomes, write_outcomes

        if any(path.resolve() == args.output.resolve() for path in args.input):
            parser.error("Output must differ from every input")
        frame, report = combine_outcomes([read_outcomes(path) for path in args.input],
                                         task_type=args.task_type,
                                         exclude_models=args.exclude_model,
                                         keep_unpriced_models=args.keep_unpriced_models)
        write_outcomes(frame, args.output)
        print(json.dumps({**report, "output": str(args.output)}, indent=2))
    elif args.command == "import-coding-tasks":
        from tine.coding_tasks import load_coding_tasks, write_coding_tasks

        if args.input.resolve() == args.output.resolve():
            parser.error("Coding task input and output must be different files")
        tasks = load_coding_tasks(args.input, dataset=args.dataset, tenant=args.tenant)
        write_coding_tasks(tasks, args.output)
        print(json.dumps({"dataset": args.dataset, "tasks": len(tasks), "output": str(args.output)}, indent=2))
    elif args.command == "preflight":
        from tine.data import read_outcomes
        from tine.pipeline import preflight as outcome_preflight

        report = outcome_preflight(read_outcomes(args.input), args.frontier,
                                   epsilon=args.epsilon, delta=args.delta, seed=args.seed)
        print(json.dumps(report, indent=2, allow_nan=False))
    elif args.command == "inspect":
        from tine.data import read_outcomes

        frame = read_outcomes(args.input)
        print(json.dumps({"rows": len(frame), "tasks": len(frame[["tenant", "task_id"]].drop_duplicates()),
                          "pending_outcomes": int(frame.success.isna().sum()),
                          "models": sorted(frame.model.unique()),
                          "tenants": sorted(frame.tenant.unique()),
                          "task_types": sorted(frame.task_type.unique()),
                          "model_coverage": frame.groupby("model").size().to_dict()}, indent=2))
    else:
        import uvicorn
        from tine.proxy import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
