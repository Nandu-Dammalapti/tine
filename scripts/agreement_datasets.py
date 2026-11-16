#!/usr/bin/env python3
"""Run the agreement check on several LLMRouterBench datasets with one fixed recipe.

For each dataset: import the recorded outcomes, pick the frontier and the cheaper
models by rules that do not look at the result, build agreement pairs and triples
with ``scripts/agreement_pairs.py``, and certify with ``tine cascade --checks``.
The rules are fixed here so that no dataset gets a model list tuned to it:

- frontier: the most accurate model with a recorded cost (ties go to the cheaper);
- cheaper models (rule ``cheapest``, the one fixed before any run): up to
  ``MAX_MODELS`` cheapest models whose mean cost is at most ``MODEL_COST_SHARE`` of
  the frontier's. Accuracy is not used.

Rule ``all-cheaper`` offers every model cheaper than the frontier instead, as the
first MMLU-Pro run did. ``agreement_pairs`` still drops groups costing more than half
of the frontier. Triples are built from the ``TRIPLE_MODELS`` cheapest of those
models only, to keep the files small. This rule was added after the ``cheapest``
results were seen.

``tine cascade`` builds its chains from eight candidates per seed. It ranks them by
what each would save on the choosing tasks if its check caught every wrong answer,
so the ranking does not look at how often a group agrees. ``--every-group`` adds a
second certification that offers every group as a one-group chain, to show what that
shortlist costs. It was also added after the first results were seen.

The shared importer refuses a dataset that has a record with a negative token count,
or a task whose prompt text differs between models. Such tasks are left out for
every model (hle has six: five and one).

No model is called and no recorded answer is executed. Per-task files stay in the
work folder; only small summary tables are written to ``--results``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import agreement_pairs  # noqa: E402
from tine.calibration import clopper_pearson_upper  # noqa: E402
from tine.cascade import run_cascade_experiment  # noqa: E402
from tine.checks import read_checks  # noqa: E402
from tine.data import read_outcomes, write_outcomes  # noqa: E402

MAX_MODELS = 8
MODEL_COST_SHARE = 0.25
EPSILONS = ("0.02", "0.04")
DELTA = 0.05  # the cascade's default, used here only to describe what can certify
CERTIFY_SHARE = 0.3  # the cascade's default share of tasks used to certify
TRIPLE_MODELS = 14
# Rule name -> (most models offered, highest cost as a share of the frontier's,
# most models offered to triples).
RULES = {"cheapest": (MAX_MODELS, MODEL_COST_SHARE, MAX_MODELS), "all-cheaper": (None, 1.0, TRIPLE_MODELS)}
# Dataset -> split folder in the archive. Datasets whose recorded "prediction" is not
# an answer (code, patches, a grader's letter, nothing) are left out; see the README.
DATASETS = {
    "arcc": "test", "medqa": "test", "winogrande": "valid", "mathbench": "test",
    "emorynlp": "test", "meld": "test", "bbh": "test", "korbench": "test",
    "finqa": "test", "math500": "test", "kandk": "test",
    "gpqa": "test", "mmlupro": "test_1000", "aime": "hybrid", "livemathbench": "test",
    "arc-agi": "v1", "hle": "test",
}
# Pooled groups of datasets that share a model pool, certified as one task mix.
# They were added after the per-dataset results, to separate "too few tasks to
# certify" from "the check is too weak". "letters" are the sets whose recorded answer
# is one option letter.
POOLS = {
    "pool-8b": ["arcc", "medqa", "winogrande", "mathbench", "emorynlp", "meld", "bbh", "korbench", "finqa",
                "math500", "kandk"],
    "pool-8b-letters": ["arcc", "medqa", "winogrande", "mathbench", "emorynlp", "meld"],
    "pool-wide": ["gpqa", "mmlupro", "aime", "livemathbench"],
    "pool-wide-small": ["gpqa", "aime", "livemathbench"],
}
# The split of the first agreement result (results/2026-11-11-agreement-check), to
# check that this driver gives the published numbers back. Not a new dataset.
REFERENCE = {"mmlupro-3000": {"mmlupro": "test_3000"}}


def sources_for(name: str) -> dict[str, str]:
    """Dataset -> split for a single dataset, a pool or the reference run."""
    if name in REFERENCE:
        return dict(REFERENCE[name])
    return {dataset: DATASETS[dataset] for dataset in POOLS.get(name, [name])}


def model_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and mean cost per model, cheapest first."""
    table = outcomes.groupby("model").agg(accuracy=("success", "mean"), mean_cost_usd=("cost_usd", "mean"))
    return table.sort_values(["mean_cost_usd", "accuracy"], ascending=[True, False])


def pick_frontier(outcomes: pd.DataFrame) -> str:
    """The most accurate model; of several equally accurate ones, the cheapest."""
    table = model_table(outcomes)
    best = table[table.accuracy == table.accuracy.max()]
    return best.mean_cost_usd.idxmin()


def pick_models(outcomes: pd.DataFrame, frontier: str, *, max_models: int | None = MAX_MODELS,
                cost_share: float = MODEL_COST_SHARE) -> list[str]:
    """The cheapest models costing at most ``cost_share`` of the frontier, by cost
    alone. Choosing by accuracy would be choosing after seeing the result."""
    table = model_table(outcomes)
    limit = cost_share * table.mean_cost_usd[frontier]
    cheap = table[(table.index != frontier) & (table.mean_cost_usd <= limit)]
    return list(cheap.index[:max_models])


def passable_loss_rate(certify_tasks: int, epsilon: float, delta: float = DELTA) -> float | None:
    """The highest observed loss rate on ``certify_tasks`` tasks whose one-sided
    Clopper-Pearson bound is still at most ``epsilon``. ``None`` when even zero
    losses cannot certify. It shows how much room a dataset of that size leaves."""
    passing = None
    for losses in range(certify_tasks + 1):
        if clopper_pearson_upper(losses, certify_tasks, delta) > epsilon:
            break
        passing = losses
    return None if passing is None else passing / certify_tasks


def run_label(name: str, max_tasks: int | None = None, sample_seed: int = 0) -> str:
    """The name a run is reported under: the dataset, plus the cut when it is sampled."""
    return name if max_tasks is None else f"{name}-{max_tasks}-sample{sample_seed}"


def latest_runs(split_dir: Path) -> list[Path]:
    """The last run file of every model folder in one split, as the importer picks it."""
    folders = sorted(path for path in Path(split_dir).iterdir() if path.is_dir())
    return [sorted(folder.glob("*.json"))[-1] for folder in folders if any(folder.glob("*.json"))]


def _negative(value) -> bool:
    try:
        return float(value) < 0
    except (TypeError, ValueError):
        return False


def refused_tasks(split_dir: Path) -> set[str]:
    """Record indexes the shared importer refuses, taken over every model's latest
    run: a negative token count, or a prompt that is not the same text for every
    model. One such record makes the importer refuse the whole dataset."""
    bad, prompts = set(), {}
    for run_file in latest_runs(split_dir):
        for record in json.loads(run_file.read_bytes())["records"]:
            index = str(record["index"])
            if any(_negative(record.get(column)) for column in ("prompt_tokens", "completion_tokens")):
                bad.add(index)
            prompt = record.get("origin_query") or record.get("prompt")
            if prompts.setdefault(index, prompt) != prompt:
                bad.add(index)
    return bad


def clean_copy(split_dir: Path, target: Path, bad: set[str]) -> None:
    """Copy each model's latest run to ``target`` without the tasks in ``bad``. A task
    is left out for every model, so that all models keep the same tasks."""
    for run_file in latest_runs(split_dir):
        data = json.loads(run_file.read_bytes())
        data["records"] = [record for record in data["records"] if str(record["index"]) not in bad]
        copy = Path(target) / run_file.parent.name / run_file.name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(json.dumps(data), encoding="utf-8")


def sample_tasks(outcomes: pd.DataFrame, max_tasks: int | None, seed: int = 0) -> pd.DataFrame:
    """A seeded random sample of ``max_tasks`` tasks, every model's rows for each.
    Used to see what a result looks like with fewer graded tasks."""
    task_ids = np.array(sorted(outcomes.task_id.unique()))
    if max_tasks is None or max_tasks >= len(task_ids):
        return outcomes
    keep = np.random.default_rng(seed).choice(task_ids, size=max_tasks, replace=False)
    return outcomes[outcomes.task_id.isin(set(keep))].reset_index(drop=True)


def group_rates(outcomes: pd.DataFrame, answers: dict, frontier: str, groups: list[tuple[str, ...]]) -> pd.DataFrame:
    """Per group, on all tasks: how often its members agree, how often an agreed
    answer is right, and how often accepting it loses or gains a task against the
    frontier. A group whose agreeing members are graded differently counts as wrong
    on that task, as ``agreement_pairs --grade-conflicts wrong`` does."""
    success = outcomes.pivot(index="task_id", columns="model", values="success").astype(bool)
    cost = outcomes.groupby("model").cost_usd.mean()
    rows = []
    for group in groups:
        agree, right, conflict = [], [], []
        graded = success[list(group)].to_numpy()
        for row, task_id in enumerate(success.index):
            given = {answers.get((task_id, model)) for model in group}
            same = len(given) == 1 and None not in given
            grades = set(graded[row].tolist())
            agree.append(same)
            conflict.append(same and len(grades) > 1)
            right.append(same and grades == {True})
        agree, right, conflict = (pd.Series(values, index=success.index) for values in (agree, right, conflict))
        frontier_right = success[frontier]
        agreed = int(agree.sum())
        all_right = success[list(group)].all(axis=1)
        rows.append({
            "group": "+".join(group), "size": len(group), "agreement_rate": agree.mean(),
            "accuracy_when_agree": right.sum() / agreed if agreed else None,
            "frontier_accuracy_when_agree": frontier_right[agree].mean() if agreed else None,
            "degradation_rate": (agree & ~right & frontier_right).mean(),
            "gain_rate": (right & ~frontier_right).mean(),
            # Every member graded right, and how often their answers are then also the
            # same string. Below 1, right answers are being spelled differently.
            "all_right_rate": all_right.mean(),
            "agree_when_all_right": (agree & all_right).sum() / all_right.sum() if all_right.any() else None,
            "grade_conflicts": int(conflict.sum()),
            "grade_conflict_share_of_agreed": conflict.sum() / agreed if agreed else None,
            "cost_share_of_frontier": sum(cost[model] for model in group) / cost[frontier],
        })
    return pd.DataFrame(rows)


def certified_rows(run_dir: Path) -> pd.DataFrame:
    """The certified picks from a cascade run, one row per epsilon, with the number
    of tasks the cascade used (it keeps one task per identical prompt)."""
    summary = pd.read_csv(Path(run_dir) / "cascade-summary.csv")
    keep = ["epsilon", "savings_mean", "savings_min", "savings_max", "accuracy_mean", "frontier_accuracy",
            "degradation_mean", "degradation_max", "over_epsilon_runs", "frontier_only_runs",
            "frontier_call_rate", "most_common_chain"]
    used = json.loads((Path(run_dir) / "cascade-summary.json").read_text())["complete_tasks"]
    return summary.loc[summary.method == "certified", keep].reset_index(drop=True).assign(tasks_used=used)


def run(*command) -> str:
    """Run one repository command and return what it printed."""
    done = subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"{' '.join(map(str, command))}\n{done.stderr[-2000:]}")
    return done.stdout


def usable_roots(sources: dict[str, str], bench: Path, work: Path) -> tuple[dict[str, Path], int]:
    """Dataset -> folder to read it from, and how many tasks were left out. A dataset
    with tasks the importer refuses is read from a copy in ``work`` without them."""
    roots, dropped = {}, 0
    for dataset, split in sources.items():
        bad = refused_tasks(bench / dataset / split)
        roots[dataset] = work / "clean" if bad else bench
        if bad:
            clean_copy(bench / dataset / split, roots[dataset] / dataset / split, bad)
            dropped += len(bad)
    return roots, dropped


def import_datasets(name: str, sources: dict[str, str], roots: dict[str, Path], work: Path) -> Path:
    """Canonical outcomes for one dataset, or several pooled into one task type,
    without the router and unpriced models. ``sources`` maps dataset to split and
    ``roots`` maps dataset to the folder that holds it."""
    raws = []
    for dataset, split in sources.items():
        raws.append(work / f"outcomes-{dataset}.jsonl")
        run("-m", "tine", "import-llmrouterbench", "--input", roots[dataset] / dataset, "--output", raws[-1],
            "--dataset", dataset, "--split", split, "--run-selection", "latest")
    combined = work / f"{name}.jsonl"
    run("-m", "tine", "combine-outcomes", "--input", *raws, "--output", combined,
        "--task-type", name, "--exclude-model", "openrouter")
    return combined


def build(outcomes, answers, frontier, models, work: Path, label: str, **options):
    """Group outcomes and checks on disk for one cascade run. Data where agreeing
    answers are graded differently is first refused, then kept with those tasks
    counted as wrong, and the report says which happened."""
    try:
        frame, checks, report = agreement_pairs.build_pairs(outcomes, answers, frontier, models=models, **options)
    except ValueError as error:
        if "Conflicting grades" not in str(error):
            raise
        frame, checks, report = agreement_pairs.build_pairs(outcomes, answers, frontier, models=models,
                                                            grade_conflicts="wrong", **options)
        report["grade_conflicts_mode"] = "wrong"
    paths = work / f"{label}.jsonl", work / f"{label}-checks.csv"
    # Prompts stay in the file: the cascade keeps one task per identical prompt and
    # splits by prompt, so dropping them would change which tasks are held out.
    write_outcomes(frame, paths[0])
    pd.DataFrame(checks).to_csv(paths[1], index=False)
    return frame, checks, report, paths


def cascade(paths, frontier: str, run_dir: Path, seeds: int) -> pd.DataFrame:
    run("-m", "tine", "cascade", "--input", paths[0], "--frontier", frontier, "--checks", paths[1],
        "--epsilon", *EPSILONS, "--seeds", seeds, "--output", run_dir)
    return certified_rows(run_dir)


def cascade_every_group(paths, frontier: str, run_dir: Path, seeds: int, groups: int) -> pd.DataFrame:
    """The same certification with every group offered, each as a one-group chain.
    ``tine cascade`` keeps eight candidates, ranked by what each would save if its
    check caught every wrong answer; that ranking ignores how often a group agrees.
    This run shows what the shortlist costs. It is a check on the method, added
    after the first results, not the method the earlier results used."""
    run_cascade_experiment(read_outcomes(paths[0]), frontier, run_dir, epsilons=[float(value) for value in EPSILONS],
                           checks=read_checks(paths[1]), seeds=range(seeds), max_length=1, top_k=groups)
    return certified_rows(run_dir)


def singles_only(frame: pd.DataFrame, checks: list[dict]):
    """From an ``--include-singles`` build, keep the single models and the frontier."""
    single = ~frame.model.str.contains("+", regex=False)
    return frame[single], [row for row in checks if "+" not in row["model"]]


def run_dataset(name: str, sources: dict[str, str], bench: Path, work: Path, seeds: int,
                rule: str = "cheapest", max_tasks: int | None = None, sample_seed: int = 0,
                every_group: bool = False) -> dict:
    """Everything for one dataset or pool (``sources``: dataset -> split): its info
    row, certified rows, group rates and model table. ``rule`` names the entry of
    ``RULES`` that picks the cheaper models; ``max_tasks`` first cuts the data to a
    random sample of that many tasks, drawn with ``sample_seed``; ``every_group``
    adds the ``cascade_every_group`` rows."""
    max_models, cost_share, triple_models = RULES[rule]
    work.mkdir(parents=True, exist_ok=True)
    label = run_label(name, max_tasks, sample_seed)
    roots, dropped = usable_roots(sources, bench, work)
    outcomes = sample_tasks(read_outcomes(import_datasets(name, sources, roots, work)), max_tasks, sample_seed)
    if outcomes.model.str.contains("+", regex=False).any():
        raise ValueError("A model name contains '+', which marks an agreement group")
    frontier = pick_frontier(outcomes)
    models = pick_models(outcomes, frontier, max_models=max_models, cost_share=cost_share)
    table = model_table(outcomes)
    info = {"dataset": label, "split": " ".join(f"{dataset}/{split}" for dataset, split in sources.items()),
            "tasks": outcomes.task_id.nunique(), "tasks_left_out": dropped,
            "priced_models": len(table),
            "frontier": frontier, "frontier_accuracy": table.accuracy[frontier],
            "frontier_mean_cost_usd": table.mean_cost_usd[frontier], "models_paired": " ".join(models),
            "models_in_triples": min(len(models), triple_models), "pairs": 0, "triples": 0, "grade_conflict_mode": "none",
            # (task, group) cases where agreeing answers were graded differently and counted wrong.
            "pairs_grade_conflicts": 0, "triples_grade_conflicts": 0}
    certified, rates = [], []
    answers = {}
    for dataset, split in sources.items():
        answers.update(agreement_pairs.recorded_answers(roots[dataset] / dataset / split, outcomes=outcomes))
    plans = [("pairs", {"group_size": 2}), ("triples", {"group_size": 3}),
             ("pairs+singles", {"group_size": 2, "include_singles": True})]
    for kind, options in plans:
        try:
            offered = models[:triple_models] if kind == "triples" else models
            frame, checks, report, paths = build(outcomes, answers, frontier, offered, work, f"{name}-{kind}", **options)
        except ValueError as error:
            if "No pair of models" not in str(error):
                raise
            continue  # fewer cheap models than the group needs
        if report.get("grade_conflicts_mode"):
            info["grade_conflict_mode"] = "wrong"
        groups = [tuple(group.split("+")) for group in report["agreement_rate"]]
        if kind in ("pairs", "triples"):
            info[kind] = len(groups)
            info[f"{kind}_grade_conflicts"] = report["grade_conflicts_counted_wrong"]
            rates.append(group_rates(outcomes, answers, frontier, groups).assign(dataset=label))
        certified.append(cascade(paths, frontier, work / f"run-{kind}", seeds).assign(dataset=label, kind=kind))
        if every_group and kind in ("pairs", "triples"):
            certified.append(cascade_every_group(paths, frontier, work / f"run-{kind}-every-group", seeds, len(groups))
                             .assign(dataset=label, kind=f"{kind}-every-group"))
        if kind == "pairs+singles":
            single_frame, single_checks = singles_only(frame, checks)
            paths = work / f"{name}-singles.jsonl", work / f"{name}-singles-checks.csv"
            write_outcomes(single_frame, paths[0])
            pd.DataFrame(single_checks).to_csv(paths[1], index=False)
            certified.append(cascade(paths, frontier, work / "run-singles", seeds).assign(dataset=label, kind="singles"))
    # A dataset with nothing to pair (emorynlp) still gets its info row and an empty table.
    certified = pd.concat(certified) if certified else pd.DataFrame(columns=["dataset", "kind"])
    rates = pd.concat(rates) if rates else pd.DataFrame(columns=["dataset", "size"])
    info.update(certify_room(info["tasks"] if certified.empty else int(certified.tasks_used.iloc[0]), rates))
    table = table.astype(float)
    return {"info": info, "certified": certified, "rates": rates,
            "models": table.assign(dataset=label, cost_share_of_frontier=table.mean_cost_usd / table.mean_cost_usd[frontier],
                                   paired=table.index.isin(models), frontier=table.index == frontier).reset_index()}


def certify_room(tasks_used: int, rates: pd.DataFrame) -> dict:
    """What a dataset of this size can certify, next to the lowest loss rate any
    group has on all of its tasks. That lowest rate is picked in hindsight, so it is
    a floor on what a run could reach, not a result."""
    certify_tasks = int(tasks_used * CERTIFY_SHARE)
    room = {"tasks_used": tasks_used, "certify_tasks": certify_tasks}
    for epsilon in EPSILONS:
        room[f"passable_loss_rate_{epsilon}"] = passable_loss_rate(certify_tasks, float(epsilon))
    for size, kind in ((2, "pair"), (3, "triple")):
        groups = rates[rates["size"] == size] if len(rates) else rates
        useful = groups[groups.agreement_rate >= 0.1] if len(groups) else groups
        room[f"lowest_{kind}_loss_rate"] = groups.degradation_rate.min() if len(groups) else None
        # Among groups that agree on at least a tenth of tasks: a group that never
        # agrees loses nothing and saves nothing.
        room[f"lowest_{kind}_loss_rate_agreeing_10pct"] = useful.degradation_rate.min() if len(useful) else None
        room[f"highest_{kind}_agreement_rate"] = groups.agreement_rate.max() if len(groups) else None
    return room


def rate_summary(rates: pd.DataFrame, certified: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and group size: the median group, and the group that the certified
    chain at the looser limit starts with most often (if any chain certified)."""
    rows = []
    for (dataset, size), groups in rates.groupby(["dataset", "size"]):
        kind = "pairs" if size == 2 else "triples"
        picks = certified[(certified.dataset == dataset) & (certified.kind == kind)].sort_values("epsilon")
        first = picks.most_common_chain.iloc[-1].split(" > ")[0] if len(picks) else ""
        picked = groups[groups.group == first]
        numeric = ["agreement_rate", "accuracy_when_agree", "frontier_accuracy_when_agree", "degradation_rate",
                   "gain_rate", "agree_when_all_right", "cost_share_of_frontier"]
        row = {"dataset": dataset, "size": size, "groups": len(groups),
               "grade_conflicts": int(groups.grade_conflicts.sum()),
               **{f"median_{column}": groups[column].median() for column in numeric},
               "picked_group": first if len(picked) else ""}
        for column in numeric:
            row[f"picked_{column}"] = picked[column].iloc[0] if len(picked) else None
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench", type=Path, required=True, help="extracted bench-release folder")
    parser.add_argument("--work", type=Path, required=True, help="folder for per-task files (not for git)")
    parser.add_argument("--results", type=Path, required=True, help="folder for the small summary tables")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=[*DATASETS, *POOLS, *REFERENCE],
                        help="datasets or pools to run (default: every single dataset)")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--label", default="", help="prefix for the summary file names, e.g. the rule")
    parser.add_argument("--brief", action="store_true", help="write only the datasets and certified tables")
    parser.add_argument("--max-tasks", type=int, help="cut each dataset to a seeded random sample of this many tasks")
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=[0],
                        help="with --max-tasks: draw one sample per seed and report each (default: 0)")
    parser.add_argument("--every-group", action="store_true",
                        help="also certify with every group offered as a one-group chain (see the module docstring)")
    parser.add_argument("--rule", choices=sorted(RULES), default="cheapest",
                        help="how the cheaper models are picked (see the module docstring)")
    args = parser.parse_args(argv)
    done = []
    for name in args.datasets:
        for sample_seed in (args.sample_seeds if args.max_tasks else [0]):
            work = args.work / run_label(name, args.max_tasks, sample_seed)
            done.append(run_dataset(name, sources_for(name), args.bench, work, args.seeds, args.rule,
                                    args.max_tasks, sample_seed, args.every_group))
    args.results.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.label}-" if args.label else ""
    certified = pd.concat(item["certified"] for item in done)
    rates = pd.concat(item["rates"] for item in done)
    first = ["dataset", "kind"]
    certified = certified[first + [column for column in certified if column not in first]]
    # Costs per task are fractions of a cent: keep more digits for them than for rates.
    cost_digits = {"frontier_mean_cost_usd": 9, "mean_cost_usd": 9}
    info = pd.DataFrame([item["info"] for item in done])
    info.round({column: cost_digits.get(column, 6) for column in info}).to_csv(
        args.results / f"{prefix}datasets.csv", index=False)
    certified.round(4).to_csv(args.results / f"{prefix}certified.csv", index=False)
    if not args.brief:
        rate_summary(rates, certified).round(4).to_csv(args.results / f"{prefix}agreement-rates.csv", index=False)
        models = pd.concat(item["models"] for item in done)
        models.round({column: cost_digits.get(column, 6) for column in models}).to_csv(
            args.results / f"{prefix}models.csv", index=False)
    rates.round(4).to_csv(args.work / "group-rates-all.csv", index=False)
    print(json.dumps([item["info"] for item in done], indent=2, default=str))


if __name__ == "__main__":
    main()
