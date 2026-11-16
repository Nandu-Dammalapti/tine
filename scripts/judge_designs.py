#!/usr/bin/env python3
"""Judge designs for code, on recorded data only.

No model is called and no patch is executed. Everything here reads the recorded
verdicts of ``claude-haiku-5-5`` and ``claude-sonnet-5-5`` on the 1,110 patches that
passed the free checks (``results/2026-11-14-api-judge``), the free checks, the
grades from our own test run and the recorded model costs.

``baseline``     rebuild the published Haiku and Sonnet chain rows with
                 ``judge_patches.py checks`` and ``tine cascade``; stop if any differs.
``tiered``       Haiku reads every patch that passed the free checks. Probability
                 >= HI accepts, <= LO rejects, anything else (including no usable
                 reply) goes to Sonnet, whose verdict decides. Haiku is paid on every
                 read, Sonnet only where consulted. Grid fixed before any chain was
                 run: HI in {0.9, 0.95}, LO in {0.1, 0.3}.
``selector``     best of k. A stage calls k cheap models, runs the free checks, the
                 judge reads every distinct patch that passed, and the stage accepts
                 the patch with the highest judge probability among those the judge
                 accepts at TAU (the judge's own verdict, or probability >= 0.9).
                 Also an oracle stage that picks a right patch whenever a member
                 wrote one, as a ceiling outside certification.
``length``       added after the selector results were seen: the sequential chain
                 with up to six stages, since a chain of three pairs reaches six models.
``calibration``  reliability, Brier, AUC, cross-validated isotonic recalibration
                 grouped by task, and catch / false-reject rates by slice.
``failures``     ``sample`` writes reading packets for 30 false accepts and 30 false
                 rejects of the Sonnet judge to a scratch folder; ``summary`` turns
                 hand-made labels into counts per category.

Every chain is certified by ``python -m tine cascade``. ``tine cascade`` gives a
candidate one cost per task, so a model that sits in two stages of one chain is paid
twice, and so is a judge call on a patch an earlier stage already judged. The
``paid-once`` columns repeat the cascade's search with each model paid once per task
and chain and each distinct patch judged once per task and chain. The same search
charged the cascade's way must reproduce the cascade's summary, or the script stops.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tine.calibration import clopper_pearson_upper  # noqa: E402
from tine.cascade import candidate_chains  # noqa: E402
from tine.checks import prepare_checks, read_checks  # noqa: E402
from tine.data import read_outcomes, validate_outcomes, write_outcomes  # noqa: E402
from tine.pipeline import complete_tasks, matrix  # noqa: E402

FRONTIER = "claude-opus-4.1"
SPLITS = {"d": (0.4, 0.3, 0.3), "c50": (0.3, 0.5, 0.2)}
EPSILONS = (0.02, 0.03, 0.04)
# Fixed before any chain was run (the task's grid).
TIERED_GRID = [(0.9, 0.1), (0.9, 0.3), (0.95, 0.1), (0.95, 0.3)]
# Fixed before any chain was run: the judge's own verdict, and probability >= 0.9.
TAUS = {"verdict": None, "p90": 0.9}
SELECTOR_SETS = {"singles": (1,), "pairs": (2,), "triples": (3,), "mixed": (1, 2, 3)}
# Fixed before the calibration tables were looked at.
FALSE_REJECT_TARGETS = (0.05, 0.10)
BUCKETS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001)
FAILURE_SEED = 20261116
FAILURE_CATEGORIES = ("partial-fix", "wrong-location", "behaviour-differs-from-tests", "judge-misread-code",
                      "judge-too-strict", "issue-ambiguous", "right-fix-different-route", "grade-questionable",
                      "other")

DEFAULTS = {
    "key": ROOT / "artifacts/judge/key-277.csv",
    "cases": ROOT / "artifacts/judge/cases-277.jsonl",
    "haiku": ROOT / "artifacts/judge/verdicts-haiku55-277.csv",
    "sonnet": ROOT / "artifacts/judge/verdicts-sonnet55-277.csv",
    "free_checks": ROOT / "artifacts/swe/repo-test-checks.csv",
    "outcomes": ROOT / "artifacts/swe/swe-274-own.jsonl",
    "dataset": ROOT / "artifacts/swe/verified/dataset.jsonl",
    "published": ROOT / "results/2026-11-14-api-judge/cascade-judge-variants.csv",
    "work": ROOT / "artifacts/judge-designs",
    "results": ROOT / "results/2026-11-16-judge-designs",
}


# ---------------------------------------------------------------- reading


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_verdicts(path: Path) -> dict[str, dict]:
    """case_id -> verdict ("accept", "reject" or "unparsed"), probability (None when
    the reply had none), cost and whether a second judge was consulted."""
    rows = read_csv(path)
    if len({row["judge"] for row in rows}) != 1:
        raise ValueError(f"{path} must hold exactly one judge")
    return {row["case_id"]: {"verdict": row["verdict"],
                             "probability": float(row["probability"]) if row["probability"] not in ("", None) else None,
                             "cost": float(row.get("cost_usd") or 0.0), "consulted": False} for row in rows}


def load_key(path: Path) -> dict[str, dict]:
    """case_id -> key row with ``right`` (our run's grade) and ``writers``."""
    key = {}
    for row in read_csv(path):
        key[row["case_id"]] = {**row, "right": row["run_resolved"] == "True",
                               "writers": [model for model in row["models"].split(";") if model]}
    return key


def load_case_of(key: dict) -> dict[tuple[str, str], str]:
    """(task_id, model) -> case_id. Identical patches for one task share a case."""
    return {(row["task_id"], model): case_id for case_id, row in key.items() for model in row["writers"]}


def load_free_checks(path: Path) -> dict[tuple[str, str], bool | None]:
    parsed = {"true": True, "false": False}
    return {(row["task_id"], row["model"]): parsed.get(row["passed"].strip().lower()) for row in read_csv(path)}


def accepts(row: dict | None, tau: float | None) -> bool:
    """A judge row accepts at ``tau``: its own verdict when ``tau`` is None, else
    probability >= tau. A missing row or a reply with no probability rejects."""
    if row is None:
        return False
    if tau is None:
        return row["verdict"] == "accept"
    return row["probability"] is not None and row["probability"] >= tau


# ---------------------------------------------------------------- tiered judge


def tiered_verdicts(first: dict, second: dict, hi: float, lo: float) -> dict[str, dict]:
    """The first judge decides when it is sure, the second otherwise.

    Probability >= ``hi`` accepts and <= ``lo`` rejects on the first judge's word.
    Anything between, and a first reply with no probability, goes to the second
    judge, whose verdict and probability are used. Cost: the first judge on every
    case, the second only where consulted. A case the second judge never saw is
    left with no verdict (``unparsed``), which rejects."""
    if not 0 <= lo < hi <= 1:
        raise ValueError("need 0 <= lo < hi <= 1")
    table = {}
    for case_id, row in first.items():
        p = row["probability"]
        if p is not None and p >= hi:
            table[case_id] = {"verdict": "accept", "probability": p, "cost": row["cost"], "consulted": False}
        elif p is not None and p <= lo:
            table[case_id] = {"verdict": "reject", "probability": p, "cost": row["cost"], "consulted": False}
        elif case_id in second:
            other = second[case_id]
            table[case_id] = {"verdict": other["verdict"], "probability": other["probability"],
                              "cost": row["cost"] + other["cost"], "consulted": True}
        else:
            table[case_id] = {"verdict": "unparsed", "probability": None, "cost": row["cost"], "consulted": False}
    return table


def screened_verdicts(first: dict, second: dict) -> dict[str, dict]:
    """The published "haiku, then sonnet": the second judge reads only what the first
    accepted and both must accept."""
    table = {}
    for case_id, row in first.items():
        if row["verdict"] == "accept" and case_id in second:
            other = second[case_id]
            table[case_id] = {"verdict": other["verdict"], "probability": other["probability"],
                              "cost": row["cost"] + other["cost"], "consulted": True}
        else:
            table[case_id] = {**row, "verdict": "reject" if row["verdict"] == "accept" else row["verdict"]}
    return table


def judge_quality(table: dict, key: dict, tau: float | None = None) -> dict:
    """Catch rate (wrong patches rejected), false-reject rate (right patches
    rejected), accepted share and precision, judge cost per check and the share of
    cases where a second judge was consulted, over every case in ``key``."""
    counts = {"right": 0, "wrong": 0, "right_rejected": 0, "wrong_rejected": 0}
    cost, consulted = 0.0, 0
    for case_id, row in key.items():
        verdict = table.get(case_id)
        kind = "right" if row["right"] else "wrong"
        counts[kind] += 1
        counts[f"{kind}_rejected"] += not accepts(verdict, tau)
        cost += verdict["cost"] if verdict else 0.0
        consulted += bool(verdict and verdict["consulted"])
    accepted_right = counts["right"] - counts["right_rejected"]
    accepted = accepted_right + counts["wrong"] - counts["wrong_rejected"]
    cases = len(key)
    return {"cases": cases, **counts,
            "catch_rate": counts["wrong_rejected"] / counts["wrong"] if counts["wrong"] else None,
            "false_reject_rate": counts["right_rejected"] / counts["right"] if counts["right"] else None,
            "accepted_share": accepted / cases, "accuracy_when_accepted": accepted_right / accepted if accepted else None,
            "second_judge_share": consulted / cases, "cost_per_check_usd": cost / cases}


# ---------------------------------------------------------------- stages


def select_patch(members: list[str], usable: dict[str, str], judge: dict, tau: float | None) -> dict:
    """One stage on one task. ``members`` are in order of preference (cheapest
    first); ``usable`` maps each member whose patch passed the free checks to its
    case id. The judge reads every distinct usable patch it has a verdict for (an
    identical patch is read once). The stage accepts the patch with the highest
    probability among those the judge accepts at ``tau``; a tie goes to the cheaper
    member. A group of one is the plain "free checks, then judge" stage."""
    cases = []
    for model in members:
        case_id = usable.get(model)
        if case_id is not None and case_id in judge and case_id not in cases:
            cases.append(case_id)
    cost = sum(judge[case_id]["cost"] for case_id in cases)
    eligible = [case_id for case_id in cases if accepts(judge[case_id], tau)]
    result = {"passed": False, "picked": None, "judge_calls": len(cases), "judge_cost_usd": cost}
    if not eligible:
        return result
    best = max(eligible, key=lambda case_id: (judge[case_id]["probability"] or 0.0, -cases.index(case_id)))
    picked = next(model for model in members if usable.get(model) == best)
    return {**result, "passed": True, "picked": picked}


def oracle_patch(members: list[str], usable: dict[str, str], right: dict[str, bool]) -> dict:
    """A stage that picks a right patch whenever a member wrote one that passed the
    free checks. No judge. A ceiling, not something that can run."""
    for model in members:
        if model in usable and right[model]:
            return {"passed": True, "picked": model, "judge_calls": 0, "judge_cost_usd": 0.0}
    return {"passed": False, "picked": None, "judge_calls": 0, "judge_cost_usd": 0.0}


def sequential_patch(members: list[str], usable: dict[str, str], judge: dict, tau: float | None,
                     own: dict[str, float]) -> dict:
    """The same members tried one at a time, cheapest first: a member is called only
    when every earlier one was rejected. Returns what ``select_patch`` returns plus
    ``model_cost`` (the members actually called). For stage-level comparison."""
    seen, cost, model_cost = set(), 0.0, 0.0
    for model in members:
        model_cost += own[model]
        case_id = usable.get(model)
        if case_id is None or case_id not in judge:
            continue
        if case_id not in seen:
            seen.add(case_id)
            cost += judge[case_id]["cost"]
        if accepts(judge[case_id], tau):
            return {"passed": True, "picked": model, "judge_calls": len(seen), "judge_cost_usd": cost,
                    "model_cost": model_cost}
    return {"passed": False, "picked": None, "judge_calls": len(seen), "judge_cost_usd": cost,
            "model_cost": model_cost}


def build_stages(outcomes, free: dict, case_of: dict, judge: dict | None, frontier: str, *,
                 sizes=(1,), tau: float | None = None, oracle: bool = False):
    """Candidates for ``tine cascade``: every group of ``sizes`` cheap models (in
    mean-cost order) whose summed mean cost is below the frontier's. A group's cost
    on a task is every member's recorded cost plus the judge's cost for every patch
    it read; its outcome is the grade of the patch it returns. A group of one keeps
    its model's name and the meaning of ``judge_patches.py checks``, except that the
    frontier's own answers are never judged or charged: a missing free check, or a
    passing patch with no case or verdict, stays unknown (and passes in the
    cascade). In a larger group such a member has no usable patch.

    Returns (outcomes frame, check rows, members per candidate)."""
    outcomes = validate_outcomes(outcomes)
    if outcomes.tenant.nunique() != 1:
        raise ValueError("Build one tenant at a time")
    if not oracle and judge is None:
        raise ValueError("A judged stage needs verdicts")
    mean_cost = outcomes.groupby("model").cost_usd.mean()
    cheap = sorted((model for model in mean_cost.index if model != frontier), key=lambda m: (mean_cost[m], m))
    by_model = {model: rows.set_index("task_id") for model, rows in outcomes.groupby("model")}
    frames, checks, members_of = [by_model[frontier].reset_index()], [], {frontier: (frontier,)}
    for size in sizes:
        for group in combinations(cheap, size):
            if sum(mean_cost[model] for model in group) >= mean_cost[frontier]:
                continue
            name = "+".join(group)
            members_of[name] = group
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
                lead = group[0]
                unknown = size == 1 and (free.get((task_id, lead)) is None or (
                    free.get((task_id, lead)) is True and not oracle and (
                        lead not in usable or usable[lead] not in judge)))
                if unknown:
                    result = {"passed": "", "picked": lead, "judge_cost_usd": 0.0}
                elif oracle:
                    right = {model: bool(by_model[model].at[task_id, "success"]) for model in group}
                    result = oracle_patch(list(group), usable, right)
                else:
                    result = select_patch(list(group), usable, judge, tau)
                picked = result["picked"]
                success.append(bool(by_model[picked].at[task_id, "success"]) if picked else False)
                extra.append(result["judge_cost_usd"])
                checks.append({"task_id": task_id, "model": name, "passed": result["passed"]})
            frame["success"] = success
            frame["cost_usd"] = cost + np.asarray(extra, dtype=float)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), checks, members_of


# ---------------------------------------------------------------- search, paid once


class Spend:
    """Per-task spend of a set of models: their own recorded costs, and the judge's
    cost and calls for the distinct patches among them that passed the free checks
    and have a verdict. Cached by model set. Stage cost charged the cascade's way is
    ``models(S) + judge(S)`` for the stage's members S; paid once, a later stage
    adds only what its members bring that earlier stages did not."""

    def __init__(self, tasks: list[str], own: dict[str, np.ndarray], free: dict, case_of: dict,
                 judge: dict | None):
        self.tasks, self.own, self.free, self.case_of, self.judge = tasks, own, free, case_of, judge or {}
        self._cache: dict[frozenset, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __call__(self, models: frozenset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if models not in self._cache:
            own = sum((self.own[m] for m in models), np.zeros(len(self.tasks)))
            cost, calls = np.zeros(len(self.tasks)), np.zeros(len(self.tasks))
            for index, task_id in enumerate(self.tasks):
                cases = {self.case_of[(task_id, m)] for m in models
                         if self.free.get((task_id, m)) is True and (task_id, m) in self.case_of}
                cases &= set(self.judge)
                cost[index] = sum(self.judge[c]["cost"] for c in cases)
                calls[index] = len(cases)
            self._cache[models] = (own, cost, calls)
        return self._cache[models]


def chain_totals(chain: list[str], table: dict, members_of: dict, spend: Spend | None, frontier: str,
                 rows, *, pay_once: bool) -> dict:
    """Savings, accuracy, degradation and judge use of one chain over ``rows``.
    ``table`` maps a candidate to per-task ``passed``, ``right`` and ``cost`` (the
    cascade's charge). With ``pay_once`` each model is paid once per task and chain
    and each distinct patch judged once per task and chain (``spend``); a task that
    reaches a stage reached every earlier one, so the earlier spend was made."""
    front = table[frontier]
    count = len(front["cost"][rows])
    reach, spent, right = np.ones(count, dtype=bool), np.zeros(count), np.zeros(count, dtype=bool)
    judge_cost, judge_calls = np.zeros(count), np.zeros(count)
    seen: frozenset = frozenset()
    for name in chain:
        stage = table[name]
        if spend is not None:
            now = seen | frozenset(members_of[name])
            own_now, cost_now, calls_now = spend(now)
            own_before, cost_before, calls_before = spend(seen)
            stage_own, stage_judge = spend(frozenset(members_of[name]))[:2]
            stage_calls = spend(frozenset(members_of[name]))[2]
            if pay_once:
                extra_own, extra_judge = own_now - own_before, cost_now - cost_before
                extra_calls = calls_now - calls_before
                seen = now
            else:
                extra_own, extra_judge, extra_calls = stage_own, stage_judge, stage_calls
            cost = (extra_own + extra_judge)[rows]
            judge_cost = judge_cost + reach * extra_judge[rows]
            judge_calls = judge_calls + reach * extra_calls[rows]
        else:
            cost = stage["cost"][rows]
        spent = spent + reach * cost
        right = right | (reach & stage["passed"][rows] & stage["right"][rows])
        reach = reach & ~stage["passed"][rows]
    spent = spent + reach * front["cost"][rows]
    right = right | (reach & front["right"][rows])
    total = front["cost"][rows].sum()
    return {"savings": float(1 - spent.sum() / total) if total else 0.0, "accuracy": float(right.mean()),
            "frontier_accuracy": float(front["right"][rows].mean()),
            "degradation_rate": float((front["right"][rows] & ~right).mean()),
            "judge_calls_per_100": float(100 * judge_calls.mean()),
            "judge_cost_pct_of_frontier": float(100 * judge_cost.sum() / total) if total else 0.0,
            "frontier_calls_per_100": float(100 * reach.mean())}


def search(frame, checks, members_of: dict, spend: Spend | None, frontier: str, *, fractions=(0.4, 0.3, 0.3),
           epsilons=EPSILONS, seeds=range(10), delta: float = 0.05, pay_once: bool = True,
           max_length: int = 3, top_k: int = 8) -> list[dict]:
    """The search of ``tine cascade`` (same parts per seed, same candidate pool, same
    uncertified pick, same fixed-sequence certified pick), with cost from
    ``chain_totals``. With ``pay_once`` False and ``spend`` None it is the cascade's
    own charge, which is how it is checked against the cascade."""
    frame, tasks, _ = complete_tasks(frame, frontier)
    models = sorted(frame.model.unique())
    frontier_index = models.index(frontier)
    outcomes = matrix(frame, tasks, models, "success", boolean=True)
    costs = matrix(frame, tasks, models, "cost_usd")
    passes = prepare_checks(pd.DataFrame(checks), tasks, models)[0]
    table = {model: {"passed": passes[:, column], "right": outcomes[:, column], "cost": costs[:, column]}
             for column, model in enumerate(models)}
    if spend is not None and list(spend.tasks) != list(tasks.task_id):
        raise ValueError("spend must cover the same tasks in the same order")
    if spend is not None:
        # The cascade's charge must be members plus judge, or the paid-once figures mean nothing.
        for name in models:
            if name != frontier:
                own, cost, _ = spend(frozenset(members_of[name]))
                if not np.allclose(own + cost, table[name]["cost"]):
                    raise ValueError(f"Candidate {name} costs differ from members plus judge")
    rows = []
    for seed in seeds:
        part = np.random.default_rng(seed).choice(3, size=len(tasks), p=list(fractions))
        choose, certify, test = (part == index for index in range(3))
        chains = [[models[index] for index in chain] for chain in
                  candidate_chains(outcomes[choose], costs[choose], frontier_index, max_length, top_k)]
        scored = [(chain, chain_totals(chain, table, members_of, spend, frontier, choose, pay_once=pay_once))
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
                lost = round(chain_totals(chain, table, members_of, spend, frontier, certify,
                                          pay_once=pay_once)["degradation_rate"] * count)
                if clopper_pearson_upper(lost, count, delta) > epsilon:
                    break  # fixed sequence: stop at the first failure
                if totals["savings"] > best:
                    picks["certified"], best = chain, totals["savings"]
            for method, chain in picks.items():
                used = [model for name in chain for model in members_of[name]]
                rows.append({"seed": seed, "epsilon": epsilon, "method": method,
                             "chain": " > ".join([*chain, frontier]), "reuses_a_model": len(used) != len(set(used)),
                             **chain_totals(chain, table, members_of, spend, frontier, test, pay_once=pay_once)})
    return rows


def summarise_search(rows: list[dict], frontier: str) -> pd.DataFrame:
    runs = pd.DataFrame(rows)
    runs["over"] = runs.degradation_rate > runs.epsilon
    return (runs.groupby(["epsilon", "method"], sort=False)
            .agg(savings_mean=("savings", "mean"), accuracy_mean=("accuracy", "mean"),
                 frontier_accuracy=("frontier_accuracy", "mean"), degradation_max=("degradation_rate", "max"),
                 over_epsilon_runs=("over", "sum"),
                 frontier_only_runs=("chain", lambda chains: int((chains == frontier).sum())),
                 reuse_runs=("reuses_a_model", "sum"),
                 judge_calls_per_100=("judge_calls_per_100", "mean"),
                 judge_cost_pct_of_frontier=("judge_cost_pct_of_frontier", "mean"),
                 frontier_calls_per_100=("frontier_calls_per_100", "mean"),
                 most_common_chain=("chain", lambda chains: chains.value_counts().index[0]))
            .reset_index())


# ---------------------------------------------------------------- running variants


class Data:
    def __init__(self, args):
        self.args = args
        self.frontier = args.frontier
        self.key = load_key(args.key)
        self.case_of = load_case_of(self.key)
        self.free = load_free_checks(args.free_checks)
        self.outcomes = read_outcomes(args.outcomes)
        self.haiku = load_verdicts(args.haiku)
        self.sonnet = load_verdicts(args.sonnet)
        own = validate_outcomes(self.outcomes).pivot(index="task_id", columns="model", values="cost_usd")
        self.own_frame = own

    def spend(self, frame, judge: dict | None) -> Spend:
        _, tasks, _ = complete_tasks(frame, self.frontier)
        ids = list(tasks.task_id)
        own = {model: self.own_frame[model].reindex(ids).to_numpy(dtype=float) for model in self.own_frame.columns}
        return Spend(ids, own, self.free, self.case_of, judge)


def cascade(folder: Path, frontier: str, fractions, seeds: int, epsilons) -> pd.DataFrame:
    out = folder / f"cascade-{'d' if tuple(fractions) == SPLITS['d'] else 'c50'}"
    if not (out / "cascade-summary.csv").exists():
        command = [sys.executable, "-m", "tine", "cascade", "--input", str(folder / "outcomes.jsonl"),
                   "--frontier", frontier, "--checks", str(folder / "checks.csv"),
                   "--epsilon", *map(str, epsilons), "--seeds", str(seeds), "--output", str(out)]
        if tuple(fractions) != SPLITS["d"]:
            command += ["--fractions", *map(str, fractions)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, cwd=ROOT)
    return pd.read_csv(out / "cascade-summary.csv")


def run_variant(data: Data, name: str, judge: dict | None, *, sizes, tau=None, oracle=False,
                labels: dict) -> list[dict]:
    """Build one candidate set, certify it with ``tine cascade`` on both splits,
    check that the search reproduces the cascade, and repeat it paid once."""
    args = data.args
    folder = args.work / "variants" / name
    frame, checks, members_of = build_stages(data.outcomes, data.free, data.case_of, judge, data.frontier,
                                             sizes=sizes, tau=tau, oracle=oracle)
    folder.mkdir(parents=True, exist_ok=True)
    write_outcomes(frame, folder / "outcomes.jsonl")
    write_rows(folder / "checks.csv", checks, ["task_id", "model", "passed"])
    frame, checks = read_outcomes(folder / "outcomes.jsonl"), read_checks(folder / "checks.csv")
    spend = data.spend(frame, None if oracle else judge)
    rows = []
    for split, fractions in SPLITS.items():
        options = {"fractions": fractions, "epsilons": tuple(args.epsilon), "seeds": range(args.seeds)}
        published = cascade(folder, data.frontier, fractions, args.seeds, args.epsilon)
        charged = summarise_search(search(frame, checks, members_of, None, data.frontier, pay_once=False,
                                          **options), data.frontier)
        for column in ("savings_mean", "accuracy_mean", "degradation_max", "frontier_only_runs"):
            if not np.allclose(charged[column].to_numpy(dtype=float), published[column].to_numpy(dtype=float)):
                raise ValueError(f"The search does not reproduce tine cascade for {name} {split}")
        every = summarise_search(search(frame, checks, members_of, spend, data.frontier, pay_once=False,
                                        **options), data.frontier)
        once = summarise_search(search(frame, checks, members_of, spend, data.frontier, pay_once=True,
                                       **options), data.frontier)
        for row, as_charged, as_cascade in zip(once.itertuples(), every.itertuples(), charged.itertuples()):
            rows.append({**labels, "split": split, "eps": row.epsilon, "method": row.method,
                         "saved": round(row.savings_mean, 3), "acc": round(row.accuracy_mean, 3),
                         "front": round(row.frontier_accuracy, 3), "degr_max": round(row.degradation_max, 3),
                         "over": int(row.over_epsilon_runs), "none": int(row.frontier_only_runs),
                         "saved_cascade_charge": round(as_cascade.savings_mean, 3),
                         "runs_reusing_a_model": int(row.reuse_runs),
                         "judge_calls_per_100": round(row.judge_calls_per_100, 1),
                         "judge_cost_pct_of_frontier": round(row.judge_cost_pct_of_frontier, 2),
                         "frontier_calls_per_100": round(row.frontier_calls_per_100, 1),
                         "most_common_chain": row.most_common_chain})
            assert abs(as_charged.savings_mean - as_cascade.savings_mean) < 1e-9
        print(f"done {name} {split}", file=sys.stderr)
    return rows


CHAIN_FIELDS = ["design", "judge", "tau", "set", "split", "eps", "method", "saved", "acc", "front", "degr_max",
                "over", "none", "saved_cascade_charge", "runs_reusing_a_model", "judge_calls_per_100",
                "judge_cost_pct_of_frontier", "frontier_calls_per_100", "most_common_chain"]


def tiered_tables(data: Data) -> dict[str, dict]:
    tables = {"haiku": data.haiku, "sonnet": data.sonnet,
              "haiku-then-sonnet": screened_verdicts(data.haiku, data.sonnet)}
    for hi, lo in TIERED_GRID:
        tables[f"tiered-hi{int(hi * 100)}-lo{int(lo * 100)}"] = tiered_verdicts(data.haiku, data.sonnet, hi, lo)
    return tables


def run_tiered(args) -> None:
    data = Data(args)
    tables = tiered_tables(data)
    quality = []
    for name, table in tables.items():
        q = judge_quality(table, data.key)
        quality.append({"judge": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in q.items()}})
        q90 = judge_quality(table, data.key, 0.9) if name in ("haiku", "sonnet") else None
        if q90:
            quality.append({"judge": f"{name}-p90",
                            **{k: round(v, 4) if isinstance(v, float) else v for k, v in q90.items()}})
    write_rows(args.results / "tiered-judge-quality.csv", quality)
    print(pd.DataFrame(quality).to_string(index=False))
    rows = []
    for name, table in tables.items():
        rows += run_variant(data, f"singles--{name}", table, sizes=(1,),
                            labels={"design": "tiered" if name.startswith("tiered") else "single judge",
                                    "judge": name, "tau": "verdict", "set": "singles"})
    write_rows(args.results / "cascade-tiered.csv", rows, CHAIN_FIELDS)


def run_selector(args) -> None:
    data = Data(args)
    rows = []
    for judge_name, judge in (("haiku", data.haiku), ("sonnet", data.sonnet)):
        for tau_name, tau in TAUS.items():
            for set_name, sizes in SELECTOR_SETS.items():
                rows += run_variant(data, f"{set_name}--{judge_name}--{tau_name}", judge, sizes=sizes, tau=tau,
                                    labels={"design": "sequential" if set_name == "singles" else "selector",
                                            "judge": judge_name, "tau": tau_name, "set": set_name})
    for set_name in ("singles", "pairs", "triples"):
        rows += run_variant(data, f"{set_name}--oracle", None, sizes=SELECTOR_SETS[set_name], oracle=True,
                            labels={"design": "oracle", "judge": "none", "tau": "", "set": set_name})
    write_rows(args.results / "cascade-selector.csv", rows, CHAIN_FIELDS)
    stages = selector_stages(data)
    write_rows(args.results / "selector-stages.csv", stages)
    print(pd.DataFrame(stages).to_string(index=False))


def run_length(args) -> None:
    """Added after the selector results were seen. ``tine cascade`` tries chains of
    at most three stages, so a chain of three pairs can reach six cheap models and a
    sequential chain only three. This repeats the sequential (singles) search with
    longer chains, paid once, so the selector can be compared with a plain chain that
    reaches as many models. The search is the one checked against the cascade."""
    data = Data(args)
    rows = []
    for judge_name, judge in (("haiku", data.haiku), ("sonnet", data.sonnet)):
        for tau_name, tau in TAUS.items():
            frame, checks, members_of = build_stages(data.outcomes, data.free, data.case_of, judge, data.frontier,
                                                     sizes=(1,), tau=tau)
            folder = args.work / "length" / f"{judge_name}--{tau_name}"
            folder.mkdir(parents=True, exist_ok=True)
            write_outcomes(frame, folder / "outcomes.jsonl")
            write_rows(folder / "checks.csv", checks, ["task_id", "model", "passed"])
            frame, checks = read_outcomes(folder / "outcomes.jsonl"), read_checks(folder / "checks.csv")
            spend = data.spend(frame, judge)
            for length in args.lengths:
                for split, fractions in SPLITS.items():
                    summary = summarise_search(search(frame, checks, members_of, spend, data.frontier,
                                                      fractions=fractions, epsilons=tuple(args.epsilon),
                                                      seeds=range(args.seeds), max_length=length), data.frontier)
                    for row in summary.itertuples():
                        rows.append({"judge": judge_name, "tau": tau_name, "max_length": length, "split": split,
                                     "eps": row.epsilon, "method": row.method, "saved": round(row.savings_mean, 3),
                                     "acc": round(row.accuracy_mean, 3), "front": round(row.frontier_accuracy, 3),
                                     "degr_max": round(row.degradation_max, 3), "over": int(row.over_epsilon_runs),
                                     "none": int(row.frontier_only_runs),
                                     "judge_calls_per_100": round(row.judge_calls_per_100, 1),
                                     "most_common_chain": row.most_common_chain})
                    print(f"done {judge_name} {tau_name} {length} {split}", file=sys.stderr)
    write_rows(args.results / "cascade-sequential-length.csv", rows)


def selector_stages(data: Data) -> list[dict]:
    """Outside certification, over all 274 tasks and every group of k cheap models:
    how often a stage returns a right patch and a wrong one, and what it costs, for
    the selector, for the same members tried one at a time, and for the oracle."""
    outcomes = validate_outcomes(data.outcomes)
    mean_cost = outcomes.groupby("model").cost_usd.mean()
    cheap = sorted((m for m in mean_cost.index if m != data.frontier), key=lambda m: (mean_cost[m], m))
    success = outcomes.set_index(["task_id", "model"]).success.astype(bool).to_dict()
    cost = outcomes.set_index(["task_id", "model"]).cost_usd.to_dict()
    tasks = sorted(set(outcomes.task_id))
    frontier_cost = np.mean([cost[(t, data.frontier)] for t in tasks])
    rows = []
    for judge_name, judge in (("haiku", data.haiku), ("sonnet", data.sonnet)):
        for tau_name, tau in TAUS.items():
            for size in (1, 2, 3):
                totals = {key: 0.0 for key in ("oracle", "sel_right", "sel_wrong", "sel_cost", "sel_calls",
                                               "seq_right", "seq_wrong", "seq_cost", "seq_calls")}
                count = 0
                for group in combinations(cheap, size):
                    for task_id in tasks:
                        usable = {m: data.case_of[(task_id, m)] for m in group
                                  if data.free.get((task_id, m)) is True and (task_id, m) in data.case_of}
                        right = {m: success[(task_id, m)] for m in group}
                        own = {m: cost[(task_id, m)] for m in group}
                        count += 1
                        totals["oracle"] += oracle_patch(list(group), usable, right)["passed"]
                        sel = select_patch(list(group), usable, judge, tau)
                        seq = sequential_patch(list(group), usable, judge, tau, own)
                        for prefix, result, models in (("sel", sel, sum(own.values())),
                                                       ("seq", seq, seq["model_cost"])):
                            if result["passed"]:
                                totals[f"{prefix}_{'right' if right[result['picked']] else 'wrong'}"] += 1
                            totals[f"{prefix}_cost"] += models + result["judge_cost_usd"]
                            totals[f"{prefix}_calls"] += result["judge_calls"]
                share = {key: value / count for key, value in totals.items()}
                rows.append({
                    "judge": judge_name, "tau": tau_name, "k": size, "groups": count // len(tasks),
                    "group_tasks": count, "oracle_right": round(share["oracle"], 4),
                    "select_right": round(share["sel_right"], 4), "select_wrong": round(share["sel_wrong"], 4),
                    "select_precision": round(share["sel_right"] / (share["sel_right"] + share["sel_wrong"]), 4)
                    if share["sel_right"] + share["sel_wrong"] else "",
                    "select_gap_closed": round(share["sel_right"] / share["oracle"], 4) if share["oracle"] else "",
                    "select_cost_pct_of_frontier": round(100 * share["sel_cost"] / frontier_cost, 2),
                    "select_judge_calls": round(share["sel_calls"], 3),
                    "sequential_right": round(share["seq_right"], 4), "sequential_wrong": round(share["seq_wrong"], 4),
                    "sequential_precision": round(share["seq_right"] / (share["seq_right"] + share["seq_wrong"]), 4)
                    if share["seq_right"] + share["seq_wrong"] else "",
                    "sequential_gap_closed": round(share["seq_right"] / share["oracle"], 4) if share["oracle"] else "",
                    "sequential_cost_pct_of_frontier": round(100 * share["seq_cost"] / frontier_cost, 2),
                    "sequential_judge_calls": round(share["seq_calls"], 3)})
    return rows


# ---------------------------------------------------------------- baseline


def run_baseline(args) -> None:
    """Rebuild the 24 published Haiku and Sonnet rows the published way (the judge's
    cost is also added to the frontier's own answers, as ``judge_patches.py checks``
    does) and compare."""
    from judge_patches import chain_inputs

    published = pd.read_csv(args.published)
    rows = []
    for name, path in (("haiku", args.haiku), ("sonnet", args.sonnet)):
        folder = args.work / "baseline" / name
        folder.mkdir(parents=True, exist_ok=True)
        chain_inputs(args.key, [path], args.free_checks, args.outcomes, folder / "checks.csv",
                     folder / "outcomes.jsonl")
        for split, fractions in SPLITS.items():
            summary = cascade(folder, args.frontier, fractions, args.seeds, args.epsilon)
            for row in summary.itertuples():
                mine = {"saved": round(row.savings_mean, 3), "acc": round(row.accuracy_mean, 3),
                        "front": round(row.frontier_accuracy, 3), "degr_max": round(row.degradation_max, 3),
                        "over": int(row.over_epsilon_runs), "none": int(row.frontier_only_runs)}
                match = published[(published.judge == name) & (published.split == split)
                                  & np.isclose(published.eps, row.epsilon) & (published.method == row.method)]
                if len(match) != 1:
                    raise ValueError(f"No published row for {name} {split} {row.epsilon} {row.method}")
                theirs = match.iloc[0]
                same = all(np.isclose(float(theirs[k]), v) for k, v in mine.items())
                rows.append({"judge": name, "split": split, "eps": row.epsilon, "method": row.method, **mine,
                             **{f"published_{k}": theirs[k] for k in mine}, "same": same})
    write_rows(args.results / "baseline-reproduction.csv", rows)
    bad = [row for row in rows if not row["same"]]
    print(f"{len(rows) - len(bad)} of {len(rows)} published rows reproduced")
    if bad:
        raise SystemExit(f"Baseline differs: {bad}")


# ---------------------------------------------------------------- calibration


def task_folds(groups, folds: int, seed: int) -> np.ndarray:
    """Fold index per row, with every row of one task in the same fold. Tasks are
    shuffled with ``seed`` and dealt round robin, so folds differ by at most one task."""
    unique = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of = {unique[index]: position % folds for position, index in enumerate(order)}
    return np.array([fold_of[group] for group in groups])


def pick_threshold(p, y, target: float) -> float:
    """Accept when probability >= the returned value. The largest threshold whose
    false-reject rate on these rows (right patches below it, or with no probability)
    is at most ``target``; that rejects the most wrong patches within the target.
    If none meets it, the most lenient one."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
    values = np.unique(p[~np.isnan(p)])
    if not len(values):
        return np.inf
    right = p[y]
    best = values[0]
    for value in values:
        if np.mean(~(right >= value)) <= target:
            best = value
        else:
            break
    return float(best)


def rates(p, y, threshold: float) -> tuple[float, float]:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
    rejected = ~(p >= threshold)
    return float(rejected[~y].mean()), float(rejected[y].mean())


def isotonic_cv(p, y, groups, folds: int = 5, seed: int = 0, targets=FALSE_REJECT_TARGETS):
    """Held-out recalibrated probabilities, and catch / false-reject rates at each
    false-reject target with the threshold chosen on the training folds, for the raw
    and the recalibrated probabilities. Folds are by task. Rows with no probability
    are left out of the fit and are always rejected."""
    from sklearn.isotonic import IsotonicRegression

    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
    fold = task_folds(groups, folds, seed)
    calibrated = np.full(len(p), np.nan)
    decisions = {(scores, target): np.zeros(len(p), dtype=bool) for scores in ("raw", "isotonic-cv")
                 for target in targets}
    for index in range(folds):
        train, test = fold != index, fold == index
        fit = train & ~np.isnan(p)
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p[fit], y[fit])
        trained = np.full(len(p), np.nan)
        trained[fit] = model.predict(p[fit])
        held = test & ~np.isnan(p)
        calibrated[held] = model.predict(p[held])
        for target in targets:
            for scores, values in (("raw", p), ("isotonic-cv", trained)):
                threshold = pick_threshold(values[train], y[train], target)
                test_values = p[test] if scores == "raw" else calibrated[test]
                decisions[(scores, target)][test] = test_values >= threshold
    table = []
    for (scores, target), accepted in decisions.items():
        table.append({"scores": scores, "target_false_reject": target,
                      "catch_rate": float((~accepted)[~y].mean()), "false_reject_rate": float((~accepted)[y].mean())})
    return calibrated, table


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def expected_calibration_error(p, y, edges=BUCKETS) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bucket = np.digitize(p, edges[1:-1])
    return float(sum(abs(p[bucket == b].mean() - y[bucket == b].mean()) * (bucket == b).mean()
                     for b in np.unique(bucket)))


def reliability(p, y, edges=BUCKETS) -> list[dict]:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (p >= low) & (p < high)
        if inside.any():
            rows.append({"bucket": f"[{low:.2f}, {min(high, 1.0):.2f}{']' if high > 1 else ')'}",
                         "cases": int(inside.sum()), "mean_probability": round(float(p[inside].mean()), 4),
                         "share_right": round(float(y[inside].mean()), 4)})
    return rows


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float | str, float | str]:
    if not total:
        return "", ""
    rate = successes / total
    centre = (rate + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return round(float(centre - half), 4), round(float(centre + half), 4)


def changed_lines(patch: str) -> int:
    """Lines a unified diff adds or removes, without file headers."""
    return sum(1 for line in patch.splitlines()
               if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))


def slice_table(judge_name: str, table: dict, key: dict, slices: dict[str, dict[str, list[str]]]) -> list[dict]:
    """Catch and false-reject rate (judge's own verdict) per slice, with 95% Wilson
    intervals and the AUC of the probability inside the slice."""
    from sklearn.metrics import roc_auc_score

    rows = []
    for dimension, values in slices.items():
        for value, cases in values.items():
            wrong = [c for c in cases if not key[c]["right"]]
            right = [c for c in cases if key[c]["right"]]
            caught = sum(not accepts(table.get(c), None) for c in wrong)
            rejected = sum(not accepts(table.get(c), None) for c in right)
            scored = [c for c in cases if table.get(c) and table[c]["probability"] is not None]
            labels = [key[c]["right"] for c in scored]
            auc = (round(float(roc_auc_score(labels, [table[c]["probability"] for c in scored])), 3)
                   if len(set(labels)) == 2 else "")
            rows.append({"judge": judge_name, "slice": dimension, "value": value, "cases": len(cases),
                         "wrong": len(wrong), "right": len(right),
                         "catch_rate": round(caught / len(wrong), 4) if wrong else "",
                         "catch_low": wilson(caught, len(wrong))[0], "catch_high": wilson(caught, len(wrong))[1],
                         "false_reject_rate": round(rejected / len(right), 4) if right else "",
                         "false_reject_low": wilson(rejected, len(right))[0],
                         "false_reject_high": wilson(rejected, len(right))[1], "auc": auc})
    return rows


def case_slices(key: dict, patches: dict[str, str]) -> tuple[dict, list[float]]:
    """Slices of the 1,110 cases: repository, the model that wrote the patch (a case
    written by several models is in each of their slices), patch size in terciles
    of changed lines, and how many models wrote the identical patch."""
    sizes = pd.Series({case_id: changed_lines(patches[case_id]) for case_id in key})
    # Sizes are whole lines, so a tercile edge is rounded down to the last size inside it.
    edges = [int(np.floor(edge)) for edge in np.quantile(sizes.to_numpy(), [1 / 3, 2 / 3])]
    slices: dict[str, dict[str, list[str]]] = {"all": {"all": list(key)}, "repository": {}, "writer": {},
                                               "patch size": {}, "identical writers": {}}
    for case_id, row in key.items():
        slices["repository"].setdefault(row["instance_id"].split("__")[0], []).append(case_id)
        for model in row["writers"]:
            slices["writer"].setdefault(model, []).append(case_id)
        size = sizes[case_id]
        label = (f"1 small (<= {edges[0]} lines)" if size <= edges[0] else
                 f"2 medium ({edges[0] + 1}-{edges[1]} lines)" if size <= edges[1] else
                 f"3 large (> {edges[1]} lines)")
        slices["patch size"].setdefault(label, []).append(case_id)
        writers = len(row["writers"])
        slices["identical writers"].setdefault("1" if writers == 1 else "2" if writers == 2 else "3+",
                                               []).append(case_id)
    for dimension in slices:
        slices[dimension] = dict(sorted(slices[dimension].items()))
    return slices, edges


def run_calibration(args) -> None:
    from sklearn.metrics import roc_auc_score

    key = load_key(args.key)
    patches = {}
    with Path(args.cases).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                case = json.loads(line)
                patches[case["case_id"]] = case["patch"]
    ids = list(key)
    y = np.array([key[c]["right"] for c in ids])
    groups = [key[c]["task_id"] for c in ids]
    slices, edges = case_slices(key, patches)
    summary, reliability_rows, target_rows, slice_rows = [], [], [], []
    for name, path in (("haiku", args.haiku), ("sonnet", args.sonnet)):
        table = load_verdicts(path)
        p = np.array([table[c]["probability"] if c in table and table[c]["probability"] is not None else np.nan
                      for c in ids], dtype=float)
        have = ~np.isnan(p)
        runs = [isotonic_cv(p, y, groups, args.folds, seed) for seed in range(args.repeats)]
        calibrated = runs[0][0]
        for scores, values in (("raw", p), ("isotonic-cv", calibrated)):
            summary.append({"judge": name, "scores": scores, "cases": len(ids), "with_probability": int(have.sum()),
                            "brier": round(brier(values[have], y[have]), 4),
                            "auc": round(float(roc_auc_score(y[have], values[have])), 4),
                            "ece": round(expected_calibration_error(values[have], y[have]), 4)})
            for row in reliability(values[have], y[have]):
                reliability_rows.append({"judge": name, "scores": scores, **row})
        verdict = np.array([accepts(table.get(c), None) for c in ids])
        target_rows.append({"judge": name, "scores": "own verdict", "target_false_reject": "",
                            "catch_rate": round(float((~verdict)[~y].mean()), 4),
                            "false_reject_rate": round(float((~verdict)[y].mean()), 4), "repeats": 1})
        frame = pd.DataFrame([row for run in runs for row in run[1]])
        for (scores, target), group in frame.groupby(["scores", "target_false_reject"], sort=False):
            target_rows.append({"judge": name, "scores": scores, "target_false_reject": target,
                                "catch_rate": round(group.catch_rate.mean(), 4),
                                "false_reject_rate": round(group.false_reject_rate.mean(), 4),
                                "repeats": len(group)})
        slice_rows += slice_table(name, table, key, slices)
    write_rows(args.results / "calibration-summary.csv", summary)
    write_rows(args.results / "calibration-reliability.csv", reliability_rows)
    write_rows(args.results / "calibration-target.csv", target_rows)
    write_rows(args.results / "judge-slices.csv", slice_rows)
    for rows in (summary, target_rows, reliability_rows, slice_rows):
        print(pd.DataFrame(rows).to_string(index=False))
    print("patch size tercile edges (changed lines):", edges)


# ---------------------------------------------------------------- failure reading


def sample_failures(key: dict, table: dict, count: int, seed: int) -> dict[str, list[str]]:
    """Case ids of ``count`` false accepts (accepted, wrong) and ``count`` false
    rejects (rejected with a reason, right), drawn with ``seed``. Replies with no
    usable verdict are not drawn: they have no reason to read."""
    accepted_wrong = sorted(c for c, row in key.items() if not row["right"] and accepts(table.get(c), None))
    rejected_right = sorted(c for c, row in key.items()
                            if row["right"] and table.get(c, {}).get("verdict") == "reject")
    rng = np.random.default_rng(seed)
    return {"false-accept": [accepted_wrong[i] for i in sorted(rng.choice(len(accepted_wrong), count, replace=False))],
            "false-reject": [rejected_right[i] for i in sorted(rng.choice(len(rejected_right), count, replace=False))]}


def run_failures_sample(args) -> None:
    key = load_key(args.key)
    verdicts = {row["case_id"]: row for row in read_csv(args.sonnet)}
    table = load_verdicts(args.sonnet)
    cases = {}
    with Path(args.cases).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                case = json.loads(line)
                cases[case["case_id"]] = case
    reference = {}
    with Path(args.dataset).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                reference[row["instance_id"]] = row
    drawn = sample_failures(key, table, args.count, args.seed)
    folder = args.work / "failures" / "packets"
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for kind, ids in drawn.items():
        for number, case_id in enumerate(ids, 1):
            packet = f"{'fa' if kind == 'false-accept' else 'fr'}-{number:02d}"
            row, ref, verdict = key[case_id], reference[key[case_id]["instance_id"]], verdicts[case_id]
            text = [f"# {packet}: {kind}", "",
                    f"Judge (claude-sonnet-5-5) verdict: {verdict['verdict']}, probability {verdict['probability']}",
                    f"Judge's reason: {verdict['reason']}",
                    f"Grade from our test run (run_resolved): {row['run_resolved']}; "
                    f"benchmark grade: {row['benchmark_right']}",
                    f"Written by: {row['models']}", f"Repository: {cases[case_id]['repository']}", "",
                    "## Issue (what the judge saw)", "", cases[case_id]["issue"], "",
                    "## Candidate patch (what the judge saw)", "", "```diff", cases[case_id]["patch"], "```", "",
                    "## Reference fix (not shown to the judge)", "", "```diff", ref["patch"], "```", "",
                    "## Hidden tests (not shown to the judge)", "", f"FAIL_TO_PASS: {ref['FAIL_TO_PASS']}", "",
                    "```diff", ref["test_patch"], "```", ""]
            (folder / f"{packet}.md").write_text("\n".join(text), encoding="utf-8")
            index.append({"packet": packet, "kind": kind, "case_id": case_id})
    write_rows(args.work / "failures" / "index.csv", index)
    print(f"wrote {len(index)} packets to {folder}")


def run_failures_summary(args) -> None:
    """Counts per kind and category from labels CSVs (packet, kind, category, ...)."""
    labels = [row for path in args.labels for row in read_csv(path)]
    unknown = {row["category"] for row in labels} - set(FAILURE_CATEGORIES)
    if unknown:
        raise SystemExit(f"Unknown categories: {sorted(unknown)}")
    if len({row["packet"] for row in labels}) != len(labels):
        raise SystemExit("A packet is labelled twice")
    frame = pd.DataFrame(labels)
    table = []
    for kind in ("false-accept", "false-reject"):
        rows = frame[frame.kind == kind]
        for category in FAILURE_CATEGORIES:
            count = int((rows.category == category).sum())
            if count:
                table.append({"kind": kind, "count_of": "category", "value": category, "cases": count,
                              "of": len(rows)})
        if "suggested_change" in rows:
            changes = rows.suggested_change.str.split(":").str[0].str.strip()
            for change, count in changes.value_counts().sort_index().items():
                table.append({"kind": kind, "count_of": "suggested change", "value": change, "cases": int(count),
                              "of": len(rows)})
    write_rows(args.results / "failure-categories.csv", table)
    print(pd.DataFrame(table).to_string(index=False))


# ---------------------------------------------------------------- main


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(command) -> None:
        for name in ("key", "cases", "haiku", "sonnet", "free_checks", "outcomes", "dataset", "published", "work",
                     "results"):
            command.add_argument(f"--{name.replace('_', '-')}", type=Path, default=DEFAULTS[name])
        command.add_argument("--frontier", default=FRONTIER)
        command.add_argument("--epsilon", type=float, nargs="+", default=list(EPSILONS))
        command.add_argument("--seeds", type=int, default=10)

    for name, text in (("baseline", "reproduce the published Haiku and Sonnet chain rows"),
                       ("tiered", "Haiku first, Sonnet where Haiku is unsure"),
                       ("selector", "best of k cheap patches, picked by the judge"),
                       ("calibration", "reliability, recalibration and slices"),
                       ("length", "sequential chains longer than three stages (added after the selector results)")):
        command = commands.add_parser(name, help=text)
        common(command)
        if name == "length":
            command.add_argument("--lengths", type=int, nargs="+", default=[3, 4, 5, 6])
        if name == "calibration":
            command.add_argument("--folds", type=int, default=5)
            command.add_argument("--repeats", type=int, default=10)
    failures = commands.add_parser("failures", help="sample Sonnet's mistakes for reading, or count labels")
    common(failures)
    failures.add_argument("action", choices=("sample", "summary"))
    failures.add_argument("--count", type=int, default=30)
    failures.add_argument("--seed", type=int, default=FAILURE_SEED)
    failures.add_argument("--labels", type=Path, nargs="+", default=[])
    args = parser.parse_args(argv)
    if args.command == "baseline":
        run_baseline(args)
    elif args.command == "tiered":
        run_tiered(args)
    elif args.command == "selector":
        run_selector(args)
    elif args.command == "calibration":
        run_calibration(args)
    elif args.command == "length":
        run_length(args)
    elif args.action == "sample":
        run_failures_sample(args)
    else:
        run_failures_summary(args)


if __name__ == "__main__":
    main()
