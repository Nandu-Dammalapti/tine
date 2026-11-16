#!/usr/bin/env python3
"""Agreement between cheap models' patches as a check on code, next to the judge.

On exact-answer work a chain accepts an answer when two cheap models give the same
one (``scripts/agreement_pairs.py``). This script asks whether the same idea helps on
SWE-bench patches, where the check so far is "free checks, then a judge"
(``scripts/judge_patches.py checks``). It builds inputs for ``tine cascade --checks``
in which a candidate is a GROUP of cheap models that are all called:

``agree-or-judge``  two members wrote the same patch and both passed the free checks:
                    accept it with no judge call. Otherwise the judge reads the
                    cheapest member's patch that passed the free checks.
``judge-only``      the same group with no agreement rule: the judge always reads the
                    cheapest passing patch. The control that shows what agreement adds.
``agree-only``      accept only on agreement; no judge is ever called.
``agree-and-judge`` accept only when two members agree and the judge accepts the
                    agreed patch. Stricter than either check; added after the
                    ``cases`` table showed the two together are right more often.

"The same patch" is an identical patch, or with ``--similarity`` a pair whose changed
lines overlap (Jaccard) at or above that value. A group costs every member's recorded
cost on every task, because agreement needs all of them to have answered, plus the
judge's recorded cost where it was consulted. A group of one model is the existing
"free checks, then judge" candidate and matches ``judge_patches.py checks``.

The frontier is never a member: a chain does not have its answer when it checks a
cheaper one. No model is called and no patch is executed; patches are compared as text.

``build``      write group outcomes, checks and a per-task detail file.
``rates``      how often groups agree and how often an agreed patch is right.
``cases``      per patch: is it right, by whether another cheap model agrees and what
               the judge said. Shows whether agreement adds anything to the judge.
``run``        build every variant, certify each with ``python -m tine cascade`` and
               write one table of the results with judge calls per task.
``summarise``  rebuild that table from the cascade outputs of an earlier ``run``.
``paid-once``  repeat the search of an earlier ``run`` with a model that sits in two
               groups of one chain paid for once per task. ``tine cascade`` charges it
               in both, which understates what groups save. Added by the verifier
               after the results were seen.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.calibration import clopper_pearson_upper  # noqa: E402
from tine.cascade import candidate_chains  # noqa: E402
from tine.checks import prepare_checks, read_checks  # noqa: E402
from tine.data import read_outcomes, validate_outcomes, write_outcomes  # noqa: E402
from tine.pipeline import complete_tasks, matrix  # noqa: E402

MODES = ("agree-or-judge", "judge-only", "agree-only", "agree-and-judge")
DETAIL_FIELDS = ["task_id", "model", "passed", "agreed", "picked", "judge_calls", "judge_cost_usd"]
# Fixed before any chain with soft agreement was run: 0.8 is the main threshold and
# 0.5 a looser one reported next to it.
SIMILARITIES = (0.8, 0.5)
SPLITS = {"d": None, "c50": (0.3, 0.5, 0.2)}
SUMMARY_FIELDS = ["variant", "judge", "split", "eps", "method", "saved", "acc", "front", "degr_max", "over", "none",
                  "judge_calls_per_100", "agreed_accepts_per_100", "frontier_calls_per_100", "judge_cost_pct_of_frontier",
                  "most_common_chain"]
PAID_ONCE_FIELDS = ["variant", "judge", "split", "eps", "method", "saved", "acc", "front", "degr_max", "over", "none",
                    "saved_charged_every_stage", "runs_reusing_a_model_every_stage", "runs_reusing_a_model"]


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def changed_line_set(patch: str) -> frozenset[str]:
    """Lines a unified diff adds or removes, stripped, without file headers. The same
    lines ``scripts/train_verifier.py`` compares for its agreement features."""
    lines = set()
    for line in patch.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")) and line[1:].strip():
            lines.add(line[1:].strip())
    return frozenset(lines)


def jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def load_patches(key_path: Path, cases_path: Path | None = None):
    """Returns ((task_id, model) -> case_id, case_id -> changed lines). Identical
    patches for one task share a case, so equal case ids mean identical patches."""
    case_of = {}
    for row in read_csv(key_path):
        for model in row["models"].split(";"):
            if model:
                case_of[(row["task_id"], model)] = row["case_id"]
    lines = {}
    if cases_path is not None:
        with Path(cases_path).open(encoding="utf-8") as stream:
            for text in stream:
                if text.strip():
                    case = json.loads(text)
                    lines[case["case_id"]] = changed_line_set(case["patch"])
    return case_of, lines


def load_judge(verdict_path: Path, threshold: float | None = None) -> dict[str, tuple[bool, float]]:
    """case_id -> (accepted, cost). ``threshold`` accepts on probability instead of the verdict."""
    rows = read_csv(verdict_path)
    if len({row["judge"] for row in rows}) != 1:
        raise ValueError(f"{verdict_path} must hold exactly one judge")

    def accepted(row) -> bool:
        if threshold is None:
            return row["verdict"] == "accept"
        return row["probability"] not in ("", None) and float(row["probability"]) >= threshold

    return {row["case_id"]: (accepted(row), float(row.get("cost_usd") or 0)) for row in rows}


def load_free_checks(path: Path) -> dict[tuple[str, str], bool | None]:
    """(task_id, model) -> passed, with None for a check that was not run."""
    parsed = {"true": True, "false": False}
    return {(row["task_id"], row["model"]): parsed.get(row["passed"].strip().lower()) for row in read_csv(path)}


def same_patch(left: str, right: str, lines: dict, similarity: float | None) -> bool:
    if left == right:
        return True
    if similarity is None:
        return False
    return jaccard(lines.get(left, frozenset()), lines.get(right, frozenset())) >= similarity


def decide(members: list[str], usable: dict[str, str], judge: dict | None, *, mode: str,
           lines: dict | None = None, similarity: float | None = None) -> dict:
    """One group on one task. ``members`` are in the order their patches are preferred
    (cheapest first); ``usable`` maps the members whose patch passed the free checks to
    its case id. Returns passed, the member whose patch is returned, whether agreement
    accepted it, and the judge calls and cost."""
    result = {"passed": False, "picked": None, "agreed": False, "judge_calls": 0, "judge_cost_usd": 0.0}
    have = [model for model in members if model in usable]
    if mode != "judge-only":
        agreeing = {model for a, b in combinations(have, 2)
                    if same_patch(usable[a], usable[b], lines or {}, similarity) for model in (a, b)}
        if agreeing and mode != "agree-and-judge":
            picked = next(model for model in have if model in agreeing)
            return {**result, "passed": True, "picked": picked, "agreed": True}
        if mode == "agree-and-judge":
            have = [model for model in have if model in agreeing]
    if mode == "agree-only" or not have:
        return result
    if judge is None:
        raise ValueError(f"Mode {mode!r} needs judge verdicts")
    picked = have[0]
    # A patch with no verdict is rejected and costs nothing: the check fails closed.
    accepted, cost = judge.get(usable[picked], (False, 0.0))
    return {**result, "passed": accepted, "picked": picked, "judge_calls": int(usable[picked] in judge),
            "judge_cost_usd": cost if usable[picked] in judge else 0.0}


def build_groups(outcomes, free: dict, case_of: dict, judge: dict | None, frontier: str, *,
                 group_sizes=(2,), mode: str = "agree-or-judge", lines: dict | None = None,
                 similarity: float | None = None, models: list[str] | None = None):
    """Returns (outcomes frame, check rows, detail rows, report).

    Every group of ``group_sizes`` cheap models whose summed mean cost is below the
    frontier's is one candidate named ``a+b``. Members are ordered by mean recorded
    cost. A group's outcome on a task is the grade of the patch it would return and
    its cost is all members' cost plus the judge's where consulted.

    A group of one keeps the meaning of ``judge_patches.py checks``: a missing free
    check or a passing patch with no case stays unknown. In a larger group such a
    member simply has no usable patch on that task.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if similarity is not None and not 0 < similarity <= 1:
        raise ValueError("similarity must be in (0, 1]")
    if mode != "agree-only" and judge is None:
        raise ValueError(f"Mode {mode!r} needs judge verdicts")
    outcomes = validate_outcomes(outcomes)
    if outcomes.tenant.nunique() != 1:
        raise ValueError("Build agreement groups for one tenant at a time")
    if frontier not in set(outcomes.model):
        raise ValueError(f"Frontier {frontier!r} is absent from the outcomes")
    mean_cost = outcomes.groupby("model").cost_usd.mean()
    cheap = sorted((model for model in mean_cost.index
                    if model != frontier and (models is None or model in models)), key=lambda m: (mean_cost[m], m))
    by_model = {model: rows.set_index("task_id") for model, rows in outcomes.groupby("model")}
    frames, checks, detail, stats = [by_model[frontier].reset_index()], [], [], {}
    for size in group_sizes:
        if mode in ("agree-only", "agree-and-judge") and size < 2:
            raise ValueError(f"{mode} needs groups of at least two models")
        for group in combinations(cheap, size):
            if sum(mean_cost[model] for model in group) >= mean_cost[frontier]:
                continue
            name = "+".join(group)
            shared = by_model[group[0]].index
            for model in group[1:]:
                shared = shared.intersection(by_model[model].index)
            frame = by_model[group[0]].loc[shared].reset_index().copy()
            frame["model"] = name
            cost = sum(by_model[model].loc[shared].cost_usd.to_numpy() for model in group)
            for column in ("input_tokens", "output_tokens"):
                if column in frame:
                    frame[column] = sum(by_model[model].loc[shared, column].to_numpy() for model in group)
            for column in ("cached_tokens", "latency_ms"):
                if column in frame:
                    frame[column] = None
            success, extra = [], []
            for task_id in shared:
                usable = {model: case_of[(task_id, model)] for model in group
                          if free.get((task_id, model)) is True and (task_id, model) in case_of}
                unknown = size == 1 and (free.get((task_id, group[0])) is None or (
                    free.get((task_id, group[0])) is True and (
                        not usable or (judge is not None and usable[group[0]] not in judge))))
                if unknown:
                    result = {"passed": "", "picked": group[0], "agreed": False, "judge_calls": 0, "judge_cost_usd": 0.0}
                else:
                    result = decide(list(group), usable, judge, mode=mode, lines=lines, similarity=similarity)
                picked = result["picked"]
                # When the check fails the group's grade is never used; record a failure.
                success.append(bool(by_model[picked].at[task_id, "success"]) if picked else False)
                extra.append(result["judge_cost_usd"])
                checks.append({"task_id": task_id, "model": name, "passed": result["passed"]})
                detail.append({"task_id": task_id, "model": name, **result})
            frame["success"] = success
            frame["cost_usd"] = cost + pd.Series(extra, dtype=float).to_numpy()
            frames.append(frame)
            rows = detail[-len(shared):]
            stats[name] = {"tasks": len(rows), "passed": sum(row["passed"] is True for row in rows),
                           "agreed": sum(row["agreed"] for row in rows),
                           "judge_calls": sum(row["judge_calls"] for row in rows)}
    if len(frames) == 1:
        raise ValueError("No group of cheap models costs less than the frontier")
    report = {"frontier": frontier, "mode": mode, "similarity": similarity, "group_sizes": list(group_sizes),
              "cheap_models": cheap, "candidates": len(frames) - 1, "groups": stats}
    return pd.concat(frames, ignore_index=True), checks, detail, report


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def agreement_rates(outcomes, free: dict, case_of: dict, lines: dict, frontier: str, *,
                    group_sizes=(2, 3), similarities=(None, *SIMILARITIES)) -> list[dict]:
    """Per group size and similarity, over every group and task: how often a group has
    a passing patch, how often two members agree, and how often the agreed patch is
    right. ``best_group`` rows give the group that agrees most often."""
    table = []
    for size in group_sizes:
        for similarity in similarities:
            _, _, detail, report = build_groups(outcomes, free, case_of, None, frontier, group_sizes=(size,),
                                                mode="agree-only", lines=lines, similarity=similarity)
            frame = validate_outcomes(outcomes).set_index(["task_id", "model"]).success
            right = {name: 0 for name in report["groups"]}
            for row in detail:
                if row["agreed"] and bool(frame[(row["task_id"], row["picked"])]):
                    right[row["model"]] += 1
            usable = {name: 0 for name in report["groups"]}
            for row in detail:
                members = row["model"].split("+")
                usable[row["model"]] += any(free.get((row["task_id"], m)) is True and (row["task_id"], m) in case_of
                                            for m in members)
            groups = report["groups"]
            total = sum(group["tasks"] for group in groups.values())
            agreed = sum(group["agreed"] for group in groups.values())
            best = max(groups, key=lambda name: (groups[name]["agreed"], name))
            table.append({
                "group_size": size, "similarity": "identical" if similarity is None else similarity,
                "groups": len(groups), "group_tasks": total,
                "with_a_passing_patch": round(sum(usable.values()) / total, 4),
                "agreed": round(agreed / total, 4),
                "agreed_of_those_with_a_passing_patch": round(agreed / max(sum(usable.values()), 1), 4),
                "agreed_patch_right": round(sum(right.values()) / agreed, 4) if agreed else "",
                "best_group": best, "best_group_agreed": round(groups[best]["agreed"] / groups[best]["tasks"], 4),
                "best_group_agreed_right": (round(right[best] / groups[best]["agreed"], 4)
                                            if groups[best]["agreed"] else "")})
    return table


def walk_chain(chain: list[str], frontier: str, outcomes, detail: dict) -> dict:
    """Run one chain over every task in ``outcomes`` and count what it used: judge
    calls, agreement accepts and frontier calls per 100 tasks, the judge's cost as a
    percentage of always-frontier cost, with savings and accuracy. Descriptive only, since the chain was picked on some of these tasks.
    ``detail`` maps (task_id, candidate) to its detail row; an unknown check passes,
    as in ``tine cascade``."""
    table = outcomes.set_index(["task_id", "model"])
    tasks = sorted(set(outcomes.task_id))
    totals = {"cost": 0.0, "frontier_cost": 0.0, "right": 0, "judge_calls": 0, "agreed": 0, "frontier_calls": 0,
              "judge_cost": 0.0}
    for task_id in tasks:
        totals["frontier_cost"] += table.at[(task_id, frontier), "cost_usd"]
        for name in chain:
            if name == frontier:
                continue
            row = detail[(task_id, name)]
            totals["cost"] += table.at[(task_id, name), "cost_usd"]
            totals["judge_calls"] += int(row["judge_calls"])
            totals["judge_cost"] += float(row["judge_cost_usd"])
            if str(row["passed"]) in ("True", ""):
                totals["agreed"] += str(row["agreed"]) == "True"
                totals["right"] += bool(table.at[(task_id, name), "success"])
                break
        else:
            totals["frontier_calls"] += 1
            totals["cost"] += table.at[(task_id, frontier), "cost_usd"]
            totals["right"] += bool(table.at[(task_id, frontier), "success"])
    n = len(tasks)
    return {"tasks": n, "savings": 1 - totals["cost"] / totals["frontier_cost"], "accuracy": totals["right"] / n,
            "judge_calls_per_100": 100 * totals["judge_calls"] / n, "agreed_accepts_per_100": 100 * totals["agreed"] / n,
            "frontier_calls_per_100": 100 * totals["frontier_calls"] / n,
            "judge_cost_pct_of_frontier": 100 * totals["judge_cost"] / totals["frontier_cost"]}


def summarise_run(run_dir: Path, outcomes_path: Path, detail_path: Path, frontier: str, labels: dict) -> list[dict]:
    """Rows for one cascade output folder: the ten-seed summary per epsilon and method,
    plus the mean over the ten chosen chains of ``walk_chain`` on all tasks."""
    outcomes = read_outcomes(outcomes_path)
    detail = {(row["task_id"], row["model"]): row for row in read_csv(detail_path)}
    runs = pd.read_csv(Path(run_dir) / "cascade-runs.csv")
    summary = pd.read_csv(Path(run_dir) / "cascade-summary.csv")
    walked, rows = {}, []
    for row in summary.itertuples():
        chains = runs.loc[(runs.epsilon == row.epsilon) & (runs.method == row.method), "chain"]
        usage = []
        for chain in chains:
            if chain not in walked:
                walked[chain] = walk_chain(chain.split(" > "), frontier, outcomes, detail)
            usage.append(walked[chain])
        rows.append({**labels, "eps": row.epsilon, "method": row.method, "saved": round(row.savings_mean, 3),
                     "acc": round(row.accuracy_mean, 3), "front": round(row.frontier_accuracy, 3),
                     "degr_max": round(row.degradation_max, 3), "over": int(row.over_epsilon_runs),
                     "none": int(row.frontier_only_runs),
                     **{field: round(mean(u[field] for u in usage), 1) for field in
                        ("judge_calls_per_100", "agreed_accepts_per_100", "frontier_calls_per_100")},
                     "judge_cost_pct_of_frontier": round(mean(u["judge_cost_pct_of_frontier"] for u in usage), 2),
                     "most_common_chain": row.most_common_chain})
    return rows


def chain_totals(chain: list[str], table: dict, own: dict, frontier: str, rows=slice(None), *,
                 pay_once: bool = True) -> dict:
    """Savings, accuracy and degradation of one chain of candidate names over ``rows``.

    ``table`` maps a candidate (and the frontier) to per-task arrays ``passed``,
    ``right`` and ``cost``, the cost as ``build_groups`` charged it: every member plus
    the judge. ``own`` maps a model to its own cost per task. With ``pay_once`` a model
    that sits in two groups of the chain is paid for in the first one only, because the
    later group can reuse its answer. The judge is charged each time a stage consults
    it. Without ``pay_once`` every stage pays its full cost, as ``tine cascade`` does."""
    front = table[frontier]
    count = len(front["cost"][rows])
    reach, spent, right, paid = np.ones(count, dtype=bool), np.zeros(count), np.zeros(count, dtype=bool), set()
    for name in chain:
        stage = table[name]
        cost = stage["cost"][rows]
        if pay_once:
            # What the group cost beyond its members' answers is the judge.
            members = name.split("+")
            judge = cost - sum(own[model][rows] for model in members)
            cost = judge + sum(own[model][rows] for model in members if model not in paid)
            paid.update(members)
        spent = spent + reach * cost
        right = right | (reach & stage["passed"][rows] & stage["right"][rows])
        reach = reach & ~stage["passed"][rows]
    spent = spent + reach * front["cost"][rows]
    right = right | (reach & front["right"][rows])
    total = front["cost"][rows].sum()
    return {"savings": float(1 - spent.sum() / total) if total else 0.0, "accuracy": float(right.mean()),
            "frontier_accuracy": float(front["right"][rows].mean()),
            "degradation_rate": float((front["right"][rows] & ~right).mean())}


def paid_once_search(frame, checks, own, frontier: str, *, fractions=(0.4, 0.3, 0.3),
                     epsilons=(0.02, 0.03, 0.04), seeds=range(10), delta: float = 0.05,
                     pay_once: bool = True, max_length: int = 3, top_k: int = 8) -> list[dict]:
    """The search of ``tine cascade`` with a chain's cost counted per model, not per group.

    ``tine cascade`` gives a candidate one cost per task, so a chain whose groups share
    a model pays for that model's answer in each of them. A real chain would call the
    model once. This repeats the cascade's search with that fixed (see
    ``chain_totals``): the same choose / certify / test parts per seed, the same pool
    of candidates, the same uncertified pick and the same fixed-sequence certified
    pick. With ``pay_once`` False it returns the cascade's own numbers, which is the
    check that the search is the same one.

    ``frame`` and ``checks`` are the group outcomes and checks from ``build_groups``;
    ``own`` is the canonical outcomes, for each model's own cost. Returns one row per
    seed, epsilon and method."""
    frame, tasks, _ = complete_tasks(frame, frontier)
    if tasks.tenant.nunique() != 1:
        raise ValueError("Search one tenant at a time")
    models = sorted(frame.model.unique())
    frontier_index = models.index(frontier)
    outcomes = matrix(frame, tasks, models, "success", boolean=True)
    costs = matrix(frame, tasks, models, "cost_usd")
    passes = prepare_checks(pd.DataFrame(checks), tasks, models)[0]
    table = {model: {"passed": passes[:, column], "right": outcomes[:, column], "cost": costs[:, column]}
             for column, model in enumerate(models)}
    own_cost = validate_outcomes(own).pivot(index="task_id", columns="model", values="cost_usd")
    members = {model for name in models if name != frontier for model in name.split("+")}
    if members - set(own_cost.columns) or own_cost.reindex(tasks.task_id)[sorted(members)].isna().any().any():
        raise ValueError("The canonical outcomes lack a cost for some group member")
    own_cost = {model: own_cost[model].reindex(tasks.task_id).to_numpy(dtype=float) for model in members}
    rows = []
    for seed in seeds:
        part = np.random.default_rng(seed).choice(3, size=len(tasks), p=list(fractions))
        choose, certify, test = (part == index for index in range(3))
        chains = [[models[index] for index in chain] for chain in
                  candidate_chains(outcomes[choose], costs[choose], frontier_index, max_length, top_k)]
        scored = [(chain, chain_totals(chain, table, own_cost, frontier, choose, pay_once=pay_once))
                  for chain in chains]
        for epsilon in epsilons:
            picks = {"uncertified": [], "certified": []}
            best = 0.0
            for chain, totals in scored:
                if totals["degradation_rate"] <= epsilon and totals["savings"] > best:
                    picks["uncertified"], best = chain, totals["savings"]
            fits = sorted((item for item in scored
                           if item[1]["degradation_rate"] <= epsilon and item[1]["savings"] > 0),
                          key=lambda item: (item[1]["degradation_rate"], -item[1]["savings"]))
            best, count = 0.0, int(certify.sum())
            for chain, totals in fits:
                lost = round(chain_totals(chain, table, own_cost, frontier, certify,
                                          pay_once=pay_once)["degradation_rate"] * count)
                if clopper_pearson_upper(lost, count, delta) > epsilon:
                    break  # fixed sequence: stop at the first failure
                if totals["savings"] > best:
                    picks["certified"], best = chain, totals["savings"]
            for method, chain in picks.items():
                used = [model for name in chain for model in name.split("+")]
                rows.append({"seed": seed, "epsilon": epsilon, "method": method,
                             "chain": " > ".join([*chain, frontier]), "reuses_a_model": len(used) != len(set(used)),
                             **chain_totals(chain, table, own_cost, frontier, test, pay_once=pay_once)})
    return rows


def summarise_search(rows: list[dict], frontier: str) -> pd.DataFrame:
    """Per epsilon and method over the seeds, with the columns of ``cascade-summary.csv``
    that the summary table uses."""
    runs = pd.DataFrame(rows)
    runs["over"] = runs.degradation_rate > runs.epsilon
    return (runs.groupby(["epsilon", "method"], sort=False)
            .agg(savings_mean=("savings", "mean"), accuracy_mean=("accuracy", "mean"),
                 frontier_accuracy=("frontier_accuracy", "mean"), degradation_max=("degradation_rate", "max"),
                 over_epsilon_runs=("over", "sum"),
                 frontier_only_runs=("chain", lambda chains: int((chains == frontier).sum())),
                 reuse_runs=("reuses_a_model", "sum")).reset_index())


def paid_once_all(args) -> None:
    """For every variant an earlier ``run`` left under ``--work``: repeat the search
    with each model paid once per task, next to the search as the cascade charged it.
    Stops if the second does not reproduce the cascade's own summary."""
    judges = parse_judges(args.judge)
    own = read_outcomes(args.outcomes)
    rows = []
    for variant in variants():
        for judge_name in (judges if variant["needs_judge"] else ["none"]):
            folder = args.work / f"{variant['name']}--{judge_name}"
            frame, checks = read_outcomes(folder / "outcomes.jsonl"), read_checks(folder / "checks.csv")
            for split, fractions in SPLITS.items():
                options = {"fractions": fractions or (0.4, 0.3, 0.3), "epsilons": args.epsilon,
                           "seeds": range(args.seeds)}
                charged = summarise_search(paid_once_search(frame, checks, own, args.frontier, pay_once=False,
                                                            **options), args.frontier)
                cascade = pd.read_csv(folder / f"cascade-{split}" / "cascade-summary.csv")
                for column in ("savings_mean", "accuracy_mean", "degradation_max", "frontier_only_runs"):
                    if not np.allclose(charged[column].to_numpy(dtype=float), cascade[column].to_numpy(dtype=float)):
                        raise ValueError(f"The search does not reproduce tine cascade for {folder.name} {split}")
                once = summarise_search(paid_once_search(frame, checks, own, args.frontier, **options),
                                        args.frontier)
                for row, as_charged in zip(once.itertuples(), charged.itertuples()):
                    rows.append({"variant": variant["name"], "judge": judge_name, "split": split,
                                 "eps": row.epsilon, "method": row.method, "saved": round(row.savings_mean, 3),
                                 "acc": round(row.accuracy_mean, 3), "front": round(row.frontier_accuracy, 3),
                                 "degr_max": round(row.degradation_max, 3), "over": int(row.over_epsilon_runs),
                                 "none": int(row.frontier_only_runs),
                                 "saved_charged_every_stage": round(as_charged.savings_mean, 3),
                                 "runs_reusing_a_model_every_stage": int(as_charged.reuse_runs),
                                 "runs_reusing_a_model": int(row.reuse_runs)})
                print(f"done {folder.name} {split}", file=sys.stderr)
    write_rows(args.summary, PAID_ONCE_FIELDS, rows)


def parse_judges(pairs: list[str]) -> dict[str, str]:
    """NAME=VERDICTS.csv arguments as a mapping. The name becomes part of a folder
    name under ``--work``, so it must be a plain word."""
    judges = {}
    for pair in pairs:
        name, separator, path = pair.partition("=")
        if not separator or not path or not name.replace("-", "").replace("_", "").isalnum():
            raise SystemExit(f"--judge takes NAME=VERDICTS.csv with a plain name, not {pair!r}")
        judges[name] = path
    return judges


def variants() -> list[dict]:
    """Every candidate set that ``run`` certifies. ``needs_judge`` False means the
    variant is the same for every judge and is run once."""
    table = [{"name": "singles", "sizes": (1,), "mode": "agree-or-judge", "similarity": None}]
    for label, similarity in [("identical", None), *((f"soft{int(s * 100):02d}", s) for s in SIMILARITIES)]:
        for sizes_label, sizes in [("pairs", (2,)), ("triples", (3,)), ("mixed", (1, 2, 3))]:
            table.append({"name": f"{sizes_label}-agree-or-judge-{label}", "sizes": sizes,
                          "mode": "agree-or-judge", "similarity": similarity})
        for sizes_label, sizes in [("pairs", (2,)), ("triples", (3,))]:
            table.append({"name": f"{sizes_label}-agree-only-{label}", "sizes": sizes,
                          "mode": "agree-only", "similarity": similarity})
    for sizes_label, sizes in [("pairs", (2,)), ("triples", (3,))]:
        table.append({"name": f"{sizes_label}-judge-only", "sizes": sizes, "mode": "judge-only", "similarity": None})
    # Added after the first results were seen; reported as such.
    for sizes_label, sizes in [("pairs", (2,)), ("triples", (3,))]:
        table.append({"name": f"{sizes_label}-agree-and-judge-soft80", "sizes": sizes, "mode": "agree-and-judge",
                      "similarity": SIMILARITIES[0]})
    for row in table:
        row["needs_judge"] = row["mode"] != "agree-only"
    return table


def build_to_files(args, judge, sizes, mode, similarity, folder: Path, case_of=None, lines=None) -> dict:
    outcomes = read_outcomes(args.outcomes)
    if case_of is None:
        case_of, lines = load_patches(args.key, args.cases)
    frame, checks, detail, report = build_groups(outcomes, load_free_checks(args.free_checks), case_of, judge,
                                                 args.frontier, group_sizes=sizes, mode=mode, lines=lines,
                                                 similarity=similarity)
    folder.mkdir(parents=True, exist_ok=True)
    write_outcomes(frame, folder / "outcomes.jsonl")
    write_rows(folder / "checks.csv", ["task_id", "model", "passed"], checks)
    write_rows(folder / "detail.csv", DETAIL_FIELDS, detail)
    (folder / "build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def case_precision(key_path: Path, case_of: dict, lines: dict, judges: dict, frontier: str, *,
                   grade: str = "run_resolved") -> list[dict]:
    """Share of cheap models' patches that are right, by agreement and by verdict.

    A patch agrees when a different cheap model's patch for the same task (also past
    the free checks) is identical or similar enough. One row per judge, level of
    agreement and verdict; ``judge`` "none" ignores the verdict."""
    key = [row for row in read_csv(key_path) if set(filter(None, row["models"].split(";"))) - {frontier}]
    by_task: dict[str, list[dict]] = {}
    for row in key:
        row["cheap"] = set(filter(None, row["models"].split(";"))) - {frontier}
        by_task.setdefault(row["task_id"], []).append(row)
    levels = {"all patches": lambda best: True, "no agreement at 0.5": lambda best: best < SIMILARITIES[1]}
    for similarity in sorted(SIMILARITIES):
        levels[f"agreement at {similarity}"] = lambda best, s=similarity: best >= s
    levels["identical patch"] = lambda best: best >= 2
    for row in key:
        others = [jaccard(lines.get(row["case_id"], frozenset()), lines.get(other["case_id"], frozenset()))
                  for other in by_task[row["task_id"]] if other is not row and other["cheap"] - row["cheap"]]
        # 2 marks a patch that two cheap models wrote character for character.
        row["best"] = 2.0 if len(row["cheap"]) > 1 else max(others, default=0.0)
    table = []
    for judge_name, verdicts in {"none": None, **judges}.items():
        for level, member in levels.items():
            for verdict in (["any"] if verdicts is None else ["accept", "reject"]):
                rows = [row for row in key if member(row["best"]) and (
                    verdicts is None or verdicts.get(row["case_id"], (False, 0.0))[0] == (verdict == "accept"))]
                right = sum(row[grade] == "True" for row in rows)
                table.append({"judge": judge_name, "agreement": level, "verdict": verdict, "cases": len(rows),
                              "right": right, "share_right": round(right / len(rows), 4) if rows else ""})
    return table


def summarise_all(args) -> None:
    """One table from every cascade output folder that ``run`` left under ``--work``."""
    judges = parse_judges(args.judge)
    rows = []
    for variant in variants():
        for judge_name in (judges if variant["needs_judge"] else ["none"]):
            folder = args.work / f"{variant['name']}--{judge_name}"
            for split in SPLITS:
                rows += summarise_run(folder / f"cascade-{split}", folder / "outcomes.jsonl", folder / "detail.csv",
                                      args.frontier, {"variant": variant["name"], "judge": judge_name, "split": split})
    write_rows(args.summary, SUMMARY_FIELDS, rows)


def run_all(args) -> None:
    """Build each variant for each judge and certify it on both splits."""
    case_of, lines = load_patches(args.key, args.cases)
    judges = parse_judges(args.judge)
    for variant in variants():
        if args.only and variant["name"] not in args.only:
            continue
        for judge_name in (judges if variant["needs_judge"] else ["none"]):
            judge = load_judge(Path(judges[judge_name])) if variant["needs_judge"] else None
            folder = args.work / f"{variant['name']}--{judge_name}"
            build_to_files(args, judge, variant["sizes"], variant["mode"], variant["similarity"], folder,
                           case_of, lines)
            for split, fractions in SPLITS.items():
                out = folder / f"cascade-{split}"
                command = [sys.executable, "-m", "tine", "cascade", "--input", str(folder / "outcomes.jsonl"),
                           "--frontier", args.frontier, "--checks", str(folder / "checks.csv"),
                           "--epsilon", *map(str, args.epsilon), "--seeds", str(args.seeds), "--output", str(out)]
                if fractions:
                    command += ["--fractions", *map(str, fractions)]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                               cwd=Path(__file__).resolve().parents[1])
                print(f"done {folder.name} {split}", file=sys.stderr)
    summarise_all(args)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def data_arguments(command, cases_required: bool) -> None:
        command.add_argument("--key", type=Path, required=True, help="judge key CSV (case_id, task_id, models)")
        command.add_argument("--cases", type=Path, required=cases_required,
                             help="judge cases JSONL (case_id, patch); needed for soft agreement")
        command.add_argument("--free-checks", type=Path, required=True, help="CSV (task_id, model, passed)")
        command.add_argument("--outcomes", type=Path, required=True, help="canonical outcomes JSONL")
        command.add_argument("--frontier", required=True)

    build = commands.add_parser("build", help="write group outcomes and checks for tine cascade")
    data_arguments(build, cases_required=False)
    build.add_argument("--verdicts", type=Path, help="one judge's verdicts CSV; not needed for agree-only")
    build.add_argument("--threshold", type=float, help="judge accepts on probability >= this")
    build.add_argument("--mode", choices=MODES, default="agree-or-judge")
    build.add_argument("--group-size", type=int, nargs="+", default=[2])
    build.add_argument("--similarity", type=float,
                       help="Jaccard similarity of changed lines that counts as agreement; default identical patches")
    build.add_argument("--output", type=Path, required=True,
                       help="folder for outcomes.jsonl, checks.csv, detail.csv and build.json")

    rates = commands.add_parser("rates", help="how often groups agree and how often an agreed patch is right")
    data_arguments(rates, cases_required=True)
    rates.add_argument("--output", type=Path, required=True)

    cases = commands.add_parser("cases", help="share of patches that are right, by agreement and verdict")
    data_arguments(cases, cases_required=True)
    cases.add_argument("--judge", nargs="+", default=[], metavar="NAME=VERDICTS.csv")
    cases.add_argument("--output", type=Path, required=True)

    for name, text in (("run", "build every variant and certify it with tine cascade"),
                       ("summarise", "rebuild the summary table from an earlier run's outputs"),
                       ("paid-once", "repeat an earlier run's search with each model paid once per task")):
        run = commands.add_parser(name, help=text)
        data_arguments(run, cases_required=True)
        run.add_argument("--judge", nargs="+", required=True, metavar="NAME=VERDICTS.csv")
        run.add_argument("--epsilon", type=float, nargs="+", default=[0.02, 0.03, 0.04])
        run.add_argument("--seeds", type=int, default=10)
        run.add_argument("--work", type=Path, required=True, help="scratch folder for inputs and cascade outputs")
        run.add_argument("--summary", type=Path, required=True, help="summary CSV of every variant")
        run.add_argument("--only", nargs="+", help="run only these variants; the summary still covers all")

    args = parser.parse_args(argv)
    if args.command == "build":
        if args.similarity is not None and args.cases is None:
            parser.error("--similarity needs --cases")
        judge = load_judge(args.verdicts, args.threshold) if args.verdicts else None
        report = build_to_files(args, judge, tuple(args.group_size), args.mode, args.similarity, args.output)
        report.pop("groups")
        print(json.dumps(report, indent=2))
    elif args.command == "rates":
        case_of, lines = load_patches(args.key, args.cases)
        table = agreement_rates(read_outcomes(args.outcomes), load_free_checks(args.free_checks), case_of, lines,
                                args.frontier)
        write_rows(args.output, list(table[0]), table)
        print(pd.DataFrame(table).to_string(index=False))
    elif args.command == "cases":
        case_of, lines = load_patches(args.key, args.cases)
        judges = {name: load_judge(Path(path)) for name, path in parse_judges(args.judge).items()}
        table = case_precision(args.key, case_of, lines, judges, args.frontier)
        write_rows(args.output, list(table[0]), table)
        print(pd.DataFrame(table).to_string(index=False))
    elif args.command == "summarise":
        summarise_all(args)
    elif args.command == "paid-once":
        paid_once_all(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
