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

from tine.calibration import clopper_pearson_upper
from tine.pipeline import complete_tasks, matrix


def chain_metrics(chain, outcomes, costs, frontier_index: int, catch: float = 1.0,
                  passes=None) -> dict:
    """Cost and quality of ``chain`` (model column indices), ending at the frontier.

    ``outcomes`` is a boolean tasks x models matrix, ``costs`` the matching costs.
    With ``passes`` (boolean, same shape: did that answer pass the check?) the chain
    escalates exactly when a check fails, and every quantity is an observed count.
    Without it, ``catch`` gives expected values. Degradation counts tasks the
    frontier solves but the chain's final answer misses.
    """
    if passes is not None:
        passes = np.asarray(passes, dtype=bool)
        if passes.shape != np.shape(outcomes):
            raise ValueError("passes must match outcomes")
        # A wrong answer that passes ships; a failed check escalates.
        catch = 1.0
        outcomes_for_flow = passes
    else:
        outcomes_for_flow = None
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
        if outcomes_for_flow is None or model == frontier_index:
            right += reach * good
            if model != frontier_index:
                reach = reach * ~good * catch  # caught failures escalate; missed ones ship
        else:
            accepted = outcomes_for_flow[:, model]
            right += reach * (accepted & good)
            reach = reach * ~accepted
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


def select_chain(outcomes, costs, frontier_index: int, *, epsilon: float, catch: float = 1.0,
                 passes=None, chains=None):
    """Cheapest chain whose degradation rate on these tasks is at most epsilon
    (point estimate, no safety margin). Returns ``[]`` (always frontier) if none fits.
    """
    chains = candidate_chains(outcomes, costs, frontier_index) if chains is None else chains
    best, best_savings = [], 0.0
    for chain in chains:
        metrics = chain_metrics(chain, outcomes, costs, frontier_index, catch, passes)
        if metrics["degradation_rate"] <= epsilon and metrics["savings"] > best_savings:
            best, best_savings = chain, metrics["savings"]
    return best


def simulated_passes(outcomes, catch: float, rng) -> np.ndarray:
    """Check results under an assumed catch rate: right answers pass; each wrong
    answer is caught (fails the check) with probability ``catch``."""
    outcomes = np.asarray(outcomes, dtype=bool)
    caught = rng.random(outcomes.shape) < catch
    return outcomes | ~caught


def certify_chain(chains, choose, certify, frontier_index: int, *, epsilon: float,
                  delta: float):
    """Pick a chain whose degradation rate is certified <= epsilon with confidence 1 - delta.

    ``choose`` and ``certify`` are (outcomes, costs, passes) on disjoint tasks.
    Chains are ordered on ``choose`` from lowest to highest degradation (ties: more
    savings first) and tested on ``certify`` in that fixed order with a one-sided
    Clopper-Pearson bound, stopping at the first failure. Among the chains that
    passed, the one saving most on ``choose`` wins. Returns (chain, bound) or
    ([], None) for always-frontier.
    """
    scored = [(chain, chain_metrics(chain, *choose[:2], frontier_index, passes=choose[2]))
              for chain in chains]
    scored = [(c, m) for c, m in scored if m["degradation_rate"] <= epsilon and m["savings"] > 0]
    scored.sort(key=lambda cm: (cm[1]["degradation_rate"], -cm[1]["savings"]))
    n = len(certify[0])
    best, best_savings, best_bound = [], 0.0, None
    for chain, metrics in scored:
        degradations = round(chain_metrics(chain, *certify[:2], frontier_index,
                                           passes=certify[2])["degradation_rate"] * n)
        bound = clopper_pearson_upper(degradations, n, delta)
        if bound > epsilon:
            break  # fixed sequence: stop at the first failure
        if metrics["savings"] > best_savings:
            best, best_savings, best_bound = chain, metrics["savings"], bound
    return best, best_bound


def checks_matrix(checks: pd.DataFrame, tasks: pd.DataFrame, models) -> np.ndarray:
    """Measured check results (columns task_id, model, passed) as a tasks x models
    matrix. A missing or unknown result counts as a pass: no check, nothing caught."""
    known = checks.dropna(subset=["passed"])
    known = known.assign(passed=known.passed.astype(str).str.lower().isin(["true", "1"]))
    pivot = known.pivot_table(index="task_id", columns="model", values="passed", aggfunc="all")
    pivot = pivot.reindex(index=tasks.task_id.to_numpy(), columns=models)
    return pivot.fillna(True).to_numpy(dtype=bool)


def run_cascade_experiment(frame, frontier: str, output, *, epsilons=(0.02, 0.01, 0.005),
                           catches=(1.0, 0.95, 0.9), checks: pd.DataFrame | None = None,
                           seeds=range(10), fractions=(0.4, 0.3, 0.3), delta: float = 0.05,
                           max_length: int = 3, top_k: int = 8) -> dict:
    """For each seed, split tasks into choose / certify / test parts. Pick a chain
    two ways: "uncertified" (lowest cost with degradation <= epsilon on the choose
    part) and "certified" (also passes a Clopper-Pearson bound <= epsilon on the
    certify part, see ``certify_chain``). Score both on the test part.

    Checks are simulated from assumed catch rates, or taken from measured
    ``checks`` (task_id, model, passed), reported as catch "measured".
    Writes cascade-runs.csv, cascade-summary.csv, cascade-summary.json, report.md.
    """
    if not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError("fractions must be three positive shares summing to 1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame, tasks, excluded = complete_tasks(frame, frontier)
    if tasks.tenant.nunique() != 1:
        raise ValueError("Run the cascade experiment on one tenant at a time")
    models = sorted(frame.model.unique())
    frontier_index = models.index(frontier)
    outcomes = matrix(frame, tasks, models, "success", boolean=True)
    costs = matrix(frame, tasks, models, "cost_usd")
    measured = None if checks is None else checks_matrix(checks, tasks, models)
    seeds = list(seeds)
    runs = []
    for seed in seeds:
        part = np.random.default_rng(seed).choice(3, size=len(tasks), p=list(fractions))
        if len(set(part)) < 3:
            raise ValueError("Too few tasks for choose / certify / test parts")
        choose, certify, test = part == 0, part == 1, part == 2
        chains = candidate_chains(outcomes[choose], costs[choose], frontier_index, max_length, top_k)
        settings = [("measured", measured)] if measured is not None else [
            (catch, simulated_passes(outcomes, catch, np.random.default_rng(10_000 + seed)))
            for catch in catches]
        for catch, passes in settings:
            def rows(mask):
                return outcomes[mask], costs[mask], passes[mask]
            for epsilon in epsilons:
                picks = {
                    "uncertified": (select_chain(*rows(choose)[:2], frontier_index, epsilon=epsilon,
                                                 passes=rows(choose)[2], chains=chains), None),
                    "certified": certify_chain(chains, rows(choose), rows(certify), frontier_index,
                                               epsilon=epsilon, delta=delta),
                }
                for method, (chain, bound) in picks.items():
                    test_outcomes, test_costs, test_passes = rows(test)
                    metrics = chain_metrics(chain, test_outcomes, test_costs, frontier_index,
                                            passes=test_passes)
                    runs.append({"seed": seed, "catch": catch, "epsilon": epsilon, "method": method,
                                 "chain": " > ".join([models[j] for j in chain] + [frontier]),
                                 "certified_bound": bound, "test_tasks": int(test.sum()),
                                 **metrics})
    runs = pd.DataFrame(runs)
    over = (runs.degradation_rate > runs.epsilon).rename("over")
    summary = (runs.assign(over=over).groupby(["catch", "epsilon", "method"], sort=False)
               .agg(savings_mean=("savings", "mean"), savings_min=("savings", "min"),
                    savings_max=("savings", "max"), accuracy_mean=("accuracy", "mean"),
                    frontier_accuracy=("frontier_accuracy", "mean"),
                    degradation_mean=("degradation_rate", "mean"),
                    degradation_max=("degradation_rate", "max"),
                    over_epsilon_runs=("over", "sum"),
                    frontier_only_runs=("chain", lambda c: int((c == frontier).sum())),
                    frontier_call_rate=("frontier_call_rate", "mean"),
                    mean_calls=("mean_calls", "mean"),
                    most_common_chain=("chain", lambda c: c.value_counts().index[0]))
               .reset_index())
    runs.to_csv(output / "cascade-runs.csv", index=False)
    summary.to_csv(output / "cascade-summary.csv", index=False)
    result = {"frontier": frontier, "models": models, "complete_tasks": len(tasks),
              "excluded_tasks": len(excluded), "seeds": len(seeds), "fractions": list(fractions),
              "delta": delta, "summary": summary.to_dict(orient="records")}
    (output / "cascade-summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    table = summary.round(4)
    (output / "report.md").write_text("\n".join([
        "# Tine escalation-chain experiment", "",
        f"Frontier: `{frontier}`. Complete tasks: {len(tasks)}. Seeds: {len(seeds)}.",
        f"Each seed splits tasks {fractions} into choose / certify / test parts.",
        "`uncertified` picks the cheapest chain whose degradation on the choose part is <= epsilon.",
        f"`certified` also requires a one-sided Clopper-Pearson bound <= epsilon (delta={delta})",
        "on the certify part, testing chains in a fixed order and stopping at the first failure.",
        "",
        "`catch` is the share of wrong answers the check catches: simulated for numbers,",
        "observed for `measured`. Benchmark labels come from hidden tests, so catch=1.0 is an",
        "optimistic upper bound. Degradation = frontier solved the task but the chain's final",
        "answer did not. Savings are against always using the frontier, on recorded costs.", "",
        table.to_markdown(index=False) if _has_tabulate() else table.to_string(index=False),
        "",
    ]))
    return result


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True
