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
        run.add_argument("--latency-samples", type=int, default=0,
                         help="Measure warm per-request CPU routing on up to N test tasks per tenant/method (0 disables)")
        if name == "run":
            run.add_argument("--input", type=Path, required=True)
            run.add_argument("--frontier", required=True, help="Predeclared frontier baseline, not selected on test labels")
            run.add_argument("--embeddings", type=Path)
            run.add_argument("--rule", choices=("absolute", "relative"), default="absolute",
                             help="relative: compare each model with the frontier's own predicted success")
            run.add_argument("--loss", choices=("degradation", "net"), default="degradation",
                             help="net: bound overall accuracy loss, letting gains offset losses")
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
    cascade = commands.add_parser("cascade", help="Evaluate try-cheap-first escalation chains")
    cascade.add_argument("--input", type=Path, required=True)
    cascade.add_argument("--frontier", required=True)
    cascade.add_argument("--output", type=Path, required=True)
    cascade.add_argument("--epsilon", type=float, nargs="+", default=[0.02, 0.01, 0.005])
    cascade.add_argument("--catch", type=float, nargs="+", default=[1.0, 0.95, 0.9],
                         help="Assumed share of wrong answers the check catches")
    cascade.add_argument("--seeds", type=int, default=10)
    cascade.add_argument("--checks", type=Path, help="Measured check results CSV (task_id, model, passed)")
    cascade.add_argument("--fractions", type=float, nargs=3, metavar=("CHOOSE", "CERTIFY", "TEST"),
                         help="Task shares when no gate or response check is fitted (default 0.4 0.3 0.3)")
    cascade.add_argument("--delta", type=float, default=0.05)
    cascade.add_argument("--max-length", type=int, default=3)
    cascade.add_argument("--catch-rates", type=Path,
                         help="CSV (model, catch_rate): also simulate each model's own catch rate")
    cascade.add_argument("--gate", choices=("logistic", "knn", "cluster"),
                         help="Also try chains gated by this router predictor")
    cascade.add_argument("--gate-embeddings", type=Path,
                         help="NPZ of fixed prompt embeddings for the gate predictor (default: TF-IDF)")
    cascade.add_argument("--response-check", action="store_true",
                         help="Also try chains that accept an answer by its response size")
    cascade.add_argument("--response-threshold", type=float, nargs="+",
                         help="Response-score thresholds to try")
    cascade.add_argument("--fit-fractions", type=float, nargs=5,
                         metavar=("TRAIN", "SCORE", "CHOOSE", "CERTIFY", "TEST"),
                         help="Task shares when a gate or response check is fitted")
    cascade.add_argument("--gate-threshold", type=float, nargs="+",
                         help="Predicted-success thresholds to try for the gate")
    freeze = commands.add_parser("freeze-policy", help="Choose and certify one chain and write it as a policy file")
    freeze.add_argument("--input", required=True, type=Path)
    freeze.add_argument("--frontier", required=True)
    freeze.add_argument("--checks", required=True, type=Path, help="Measured check results CSV (task_id, model, passed)")
    freeze.add_argument("--epsilon", required=True, type=float)
    freeze.add_argument("--delta", type=float, default=0.05)
    freeze.add_argument("--seed", type=int, default=42)
    freeze.add_argument("--max-length", type=int, default=3)
    freeze.add_argument("--output", required=True, type=Path, help="Policy JSON")
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
                                 synthetic=args.command == "demo", latency_samples=args.latency_samples,
                                 rule=getattr(args, "rule", "absolute"),
                                 loss=getattr(args, "loss", "degradation"))
        print(json.dumps(summary, indent=2, allow_nan=False))
        print(f"Report: {args.output / 'report.md'}")
    elif args.command == "import-llmrouterbench":
        from tine.data import import_llmrouterbench

        frame = import_llmrouterbench(args.input, args.output, tenant=args.tenant,
                                     dataset=args.dataset, model=args.model, split=args.split,
                                     score_threshold=args.score_threshold, run_selection=args.run_selection)
        print(json.dumps({"rows": len(frame), "tasks": len(frame[["tenant", "task_id"]].drop_duplicates()),
                          "models": sorted(frame.model.unique()), "output": str(args.output)}, indent=2))
    elif args.command == "cascade":
        from tine.cascade import run_cascade_experiment
        from tine.checks import read_checks
        from tine.data import read_outcomes

        checks = read_checks(args.checks) if args.checks else None
        catch_rates = None
        if args.catch_rates:
            import pandas as pd
            rates = pd.read_csv(args.catch_rates, dtype={"model": str}, keep_default_na=False)
            if not {"model", "catch_rate"} <= set(rates.columns) or rates.model.duplicated().any():
                raise SystemExit("--catch-rates needs model and catch_rate columns, one row per model")
            catch_rates = dict(zip(rates.model, rates.catch_rate.astype(float)))
        result = run_cascade_experiment(read_outcomes(args.input), args.frontier, args.output,
                                        epsilons=args.epsilon, catches=args.catch, checks=checks,
                                        delta=args.delta, seeds=range(args.seeds),
                                        max_length=args.max_length, gate_method=args.gate,
                                        catch_rates=catch_rates, gate_embeddings=args.gate_embeddings,
                                        response_check=args.response_check,
                                        fit_fractions=args.fit_fractions,
                                        **({"fractions": tuple(args.fractions)} if args.fractions else {}),
                                        **({"gate_thresholds": args.gate_threshold}
                                           if args.gate_threshold else {}),
                                        **({"response_thresholds": args.response_threshold}
                                           if args.response_threshold else {}))
        print(json.dumps(result["summary"], indent=2, default=float))
        print(f"Report: {args.output / 'report.md'}")
    elif args.command == "freeze-policy":
        from tine.artifacts import atomic_write_text
        from tine.checks import read_checks
        from tine.data import read_outcomes
        from tine.policy import dumps, freeze_policy

        policy = freeze_policy(read_outcomes(args.input), args.frontier, read_checks(args.checks),
                               epsilon=args.epsilon, delta=args.delta, seed=args.seed,
                               max_length=args.max_length)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.output, dumps(policy))
        print(json.dumps({key: policy[key] for key in ("stages", "guarantee", "estimate")}, indent=2, default=float))
        print(f"Policy: {args.output}")
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
