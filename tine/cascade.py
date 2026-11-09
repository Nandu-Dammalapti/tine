"""Escalation chains on existing graded outcomes: try cheaper models in order,
check each answer, and fall back to the frontier model.

For coding, the "check" is a test run. Benchmark labels come from hidden tests,
so ``catch=1.0`` (every wrong answer is caught) is an optimistic upper bound;
``catch < 1`` lets that share of wrong answers ship as failures instead of
escalating. Real check quality must be measured before claiming results.

A chain is chosen on one random half of the tasks and scored on the other half,
repeated over several seeds, so reported numbers are held-out. No model calls.
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from tine.pipeline import complete_tasks, matrix


def chain_metrics(chain, outcomes, costs, frontier_index: int, catch: float = 1.0) -> dict:
    """Expected cost and quality of ``chain`` (model column indices), ending at the frontier.

    ``outcomes`` is a boolean tasks x models matrix, ``costs`` the matching costs.
    Degradation counts tasks the frontier solves but the chain's final answer misses.
    """
    outcomes = np.asarray(outcomes, dtype=bool)
    costs = np.asarray(costs, dtype=float)
    if outcomes.shape != costs.shape or outcomes.ndim != 2 or not len(outcomes):
        raise ValueError("outcomes and costs must be matching nonempty task x model matrices")
    if not 0 <= catch <= 1:
        raise ValueError("catch must be between 0 and 1")
    if frontier_index in chain or len(set(chain)) != len(chain):
        raise ValueError("chain must list distinct non-frontier models")
    n = len(outcomes)
    reach, right, spent, calls = np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for model in [*chain, frontier_index]:
        good = outcomes[:, model]
        spent += reach * costs[:, model]
        calls += reach
        right += reach * good
        if model != frontier_index:
            reach = reach * ~good * catch  # caught failures escalate; missed ones ship
    frontier_ok, frontier_cost = outcomes[:, frontier_index], costs[:, frontier_index]
    total = frontier_cost.sum()
    return {
        "savings": float(1 - spent.sum() / total) if total else None,
        "accuracy": float(right.mean()),
        "frontier_accuracy": float(frontier_ok.mean()),
        "degradation_rate": float((frontier_ok * (1 - right)).mean()),
        "mean_cost_usd": float(spent.mean()),
        "frontier_mean_cost_usd": float(frontier_cost.mean()),
        "frontier_call_rate": float(reach.mean()),
        "mean_calls": float(calls.mean()),
    }


def candidate_chains(outcomes, costs, frontier_index: int, max_length: int = 3, top_k: int = 8):
    """Ordered chains of up to ``max_length`` cheaper models, drawn from the ``top_k``
    models that save most when tried alone first."""
    costs = np.asarray(costs, dtype=float)
    frontier_mean = costs[:, frontier_index].mean()
    cheaper = [j for j in range(costs.shape[1])
               if j != frontier_index and costs[:, j].mean() < frontier_mean]
    ranked = sorted(cheaper, key=lambda j: -chain_metrics([j], outcomes, costs, frontier_index)["savings"])
    pool = ranked[:top_k]
    return [list(p) for length in range(1, max_length + 1) for p in permutations(pool, length)]


def select_chain(outcomes, costs, frontier_index: int, *, catch: float, epsilon: float,
                 chains=None):
    """Cheapest chain whose degradation rate on these tasks is at most epsilon.

    Returns ``[]`` (always use the frontier) when no chain fits.
    """
    chains = candidate_chains(outcomes, costs, frontier_index) if chains is None else chains
    best, best_savings = [], 0.0
    for chain in chains:
        metrics = chain_metrics(chain, outcomes, costs, frontier_index, catch)
        if metrics["degradation_rate"] <= epsilon and metrics["savings"] > best_savings:
            best, best_savings = chain, metrics["savings"]
    return best


def run_cascade_experiment(frame, frontier: str, output, *, epsilons=(0.02, 0.01, 0.005),
                           catches=(1.0, 0.95, 0.9), seeds=range(10), choose_fraction=0.5,
                           max_length: int = 3, top_k: int = 8) -> dict:
    """Choose a chain on part of the tasks and score it on the rest, for each seed,
    epsilon and catch rate. Writes cascade-runs.csv, cascade-summary.csv, report.md."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame, tasks, excluded = complete_tasks(frame, frontier)
    if tasks.tenant.nunique() != 1:
        raise ValueError("Run the cascade experiment on one tenant at a time")
    models = sorted(frame.model.unique())
    frontier_index = models.index(frontier)
    outcomes = matrix(frame, tasks, models, "success", boolean=True)
    costs = matrix(frame, tasks, models, "cost_usd")
    runs = []
    for seed in seeds:
        choose = np.random.default_rng(seed).random(len(tasks)) < choose_fraction
        if choose.all() or not choose.any():
            raise ValueError("Too few tasks to split into choose and test halves")
        chains = candidate_chains(outcomes[choose], costs[choose], frontier_index, max_length, top_k)
        for catch in catches:
            for epsilon in epsilons:
                chain = select_chain(outcomes[choose], costs[choose], frontier_index,
                                     catch=catch, epsilon=epsilon, chains=chains)
                metrics = chain_metrics(chain, outcomes[~choose], costs[~choose], frontier_index, catch)
                runs.append({"seed": seed, "catch": catch, "epsilon": epsilon,
                             "chain": " > ".join([models[j] for j in chain] + [frontier]),
                             "test_tasks": int((~choose).sum()), **metrics})
    runs = pd.DataFrame(runs)
    summary = (runs.groupby(["catch", "epsilon"], sort=False)
               .agg(savings_mean=("savings", "mean"), savings_min=("savings", "min"),
                    savings_max=("savings", "max"), accuracy_mean=("accuracy", "mean"),
                    frontier_accuracy=("frontier_accuracy", "mean"),
                    degradation_mean=("degradation_rate", "mean"),
                    degradation_max=("degradation_rate", "max"),
                    over_epsilon_runs=("degradation_rate",
                                       lambda d: int((d > runs.loc[d.index, "epsilon"]).sum())),
                    frontier_call_rate=("frontier_call_rate", "mean"),
                    mean_calls=("mean_calls", "mean"),
                    most_common_chain=("chain", lambda c: c.value_counts().index[0]))
               .reset_index())
    runs.to_csv(output / "cascade-runs.csv", index=False)
    summary.to_csv(output / "cascade-summary.csv", index=False)
    result = {"frontier": frontier, "models": models, "complete_tasks": len(tasks),
              "excluded_tasks": len(excluded), "seeds": len(list(seeds)),
              "choose_fraction": choose_fraction,
              "summary": summary.to_dict(orient="records")}
    (output / "cascade-summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    (output / "report.md").write_text("\n".join([
        "# Tine escalation-chain experiment", "",
        f"Frontier: `{frontier}`. Complete tasks: {len(tasks)}. Seeds: {len(list(seeds))}.",
        "Each seed chooses a chain on a random half of the tasks and scores it on the other half.",
        "",
        "`catch` is the assumed share of wrong answers the check catches. Benchmark labels use",
        "hidden tests, so catch=1.0 is an optimistic upper bound, not a measured result.",
        "Degradation = frontier solved the task but the chain's final answer did not.",
        "Savings are against always using the frontier, on recorded per-task costs.", "",
        summary.round(4).to_markdown(index=False) if _has_tabulate() else summary.round(4).to_string(index=False),
        "",
    ]))
    return result


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True
