"""Escalation chains on existing graded outcomes: try cheaper models in order,
check each answer, and fall back to the frontier model.

For coding, the "check" is a test run. Benchmark labels come from hidden tests,
so ``catch=1.0`` (every wrong answer is caught) is an optimistic upper bound;
``catch < 1`` lets that share of wrong answers ship as failures instead of
escalating. Real check quality must be measured before claiming results.

A chain is chosen on one random half of the tasks and scored on the other half,
repeated over several seeds, so reported numbers are held-out. No model calls.

A chain can also be gated by the router's predictor: a cheaper model is tried only
on tasks where its predicted success is at least ``tau``, so fewer wrong answers
reach the check at all. The predictor is fitted on tasks kept apart from the
choose / certify / test parts.
"""

from __future__ import annotations

import json
from itertools import permutations

import numpy as np
import pandas as pd

from tine.calibration import _integer, _matrices, _probability, clopper_pearson_upper
from tine.artifacts import experiment_output
from tine.checks import prepare_checks
from tine.pipeline import complete_tasks, embedding_fingerprint, load_embeddings, matrix
from tine.routing import METHODS, TASK_COLUMNS, Router

DEFAULT_GATE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def gate(scores, tau):
    """Tasks x models eligibility for a gated chain: a model may be tried where its
    predicted success is at least ``tau``. ``tau=None`` means no gate."""
    if tau is None:
        return None
    if scores is None:
        raise ValueError("A gate threshold requires predicted scores")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be a finite task-by-model matrix of probabilities")
    return scores >= _probability(tau, "tau")


def chain_metrics(chain, outcomes, costs, frontier_index: int, catch: float = 1.0,
                  passes=None, eligible=None) -> dict:
    """Cost and quality of ``chain`` (model column indices), ending at the frontier.

    ``outcomes`` is a boolean tasks x models matrix, ``costs`` the matching costs.
    With ``passes`` (boolean, same shape: did that answer pass the check?) the chain
    escalates exactly when a check fails, and every quantity is an observed count.
    Without it, ``catch`` gives expected values. Degradation counts tasks the
    frontier solves but the chain's final answer misses.

    With ``eligible`` (boolean, same shape, see ``gate``) a cheaper model is skipped,
    at no cost, on tasks where it is not eligible. The frontier is always eligible.
    """
    outcomes, costs, frontier_index = _matrices(outcomes, costs, frontier_index)
    if not len(outcomes):
        raise ValueError("outcomes and costs must be nonempty")
    catch = _probability(catch, "catch")
    chain = [_integer(model, "chain model index") for model in chain]
    if any(model >= outcomes.shape[1] for model in chain):
        raise ValueError("chain model index is outside the model columns")
    if passes is not None:
        passes = np.asarray(passes)
        if passes.shape != outcomes.shape or passes.dtype != np.dtype(bool):
            raise ValueError("passes must match outcomes and contain known Boolean check results")
        # A wrong answer that passes ships; a failed check escalates.
        catch = 1.0
        outcomes_for_flow = passes
    else:
        outcomes_for_flow = None
    if eligible is not None:
        eligible = np.asarray(eligible)
        if eligible.shape != outcomes.shape or eligible.dtype != np.dtype(bool):
            raise ValueError("eligible must match outcomes and contain Boolean gate decisions")
    if frontier_index in chain or len(set(chain)) != len(chain):
        raise ValueError("chain must list distinct non-frontier models")
    n = len(outcomes)
    reach, right, spent, calls = np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for model in [*chain, frontier_index]:
        good = outcomes[:, model]
        gated = eligible is not None and model != frontier_index
        tried = reach * eligible[:, model] if gated else reach
        spent += tried * costs[:, model]
        calls += tried
        if model == frontier_index:
            right += tried * good
        elif outcomes_for_flow is None:
            right += tried * good
            # caught failures escalate; right answers and missed failures ship
            reach = reach - tried * (good + ~good * (1 - catch))
        else:
            accepted = outcomes_for_flow[:, model]
            right += tried * (accepted & good)
            reach = reach - tried * accepted
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
    outcomes, costs, frontier_index = _matrices(outcomes, costs, frontier_index)
    if not len(outcomes):
        raise ValueError("Candidate selection requires nonempty outcomes")
    max_length = _integer(max_length, "max_length", minimum=1)
    top_k = _integer(top_k, "top_k", minimum=1)
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
    epsilon = _probability(epsilon, "epsilon")
    catch = _probability(catch, "catch")
    chains = candidate_chains(outcomes, costs, frontier_index) if chains is None else chains
    best, best_savings = [], 0.0
    for chain in chains:
        metrics = chain_metrics(chain, outcomes, costs, frontier_index, catch, passes)
        if (metrics["degradation_rate"] <= epsilon and metrics["savings"] is not None
                and metrics["savings"] > best_savings):
            best, best_savings = chain, metrics["savings"]
    return best


def simulated_passes(outcomes, catch, rng) -> np.ndarray:
    """Check results under an assumed catch rate: right answers pass; each wrong
    answer is caught (fails the check) with probability ``catch``. ``catch`` is one
    rate or one rate per model column."""
    outcomes = np.asarray(outcomes)
    if outcomes.dtype != np.dtype(bool) or outcomes.ndim != 2:
        raise ValueError("outcomes must be a task-by-model matrix of known Boolean labels")
    if np.ndim(catch):
        catch = np.array([_probability(value, "catch") for value in catch])
        if catch.shape != (outcomes.shape[1],):
            raise ValueError("Per-model catch rates need one rate per model column")
    else:
        catch = _probability(catch, "catch")
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
    chain, _, bound = certify_gated_chain(chains, (None,), (*choose[:3], None),
                                          (*certify[:3], None), frontier_index,
                                          epsilon=epsilon, delta=delta)
    return chain, bound


def certify_gated_chain(chains, taus, choose, certify, frontier_index: int, *,
                        epsilon: float, delta: float):
    """``certify_chain`` over (chain, tau) policies, see ``gate``.

    ``choose`` and ``certify`` are (outcomes, costs, passes, scores) on disjoint
    tasks; the scores must come from a predictor fitted on neither. Every chain is
    paired with every tau (``None`` = no gate) and the pairs go through the same
    fixed-sequence test. Returns (chain, tau, bound) or ([], None, None).
    """
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta", open_interval=True)
    taus = list(taus)
    if not taus:
        raise ValueError("Require at least one gate setting; use None for no gate")
    choose_gates = [gate(choose[3], tau) for tau in taus]
    certify_gates = [gate(certify[3], tau) for tau in taus]
    scored = [(chain, setting, chain_metrics(chain, *choose[:2], frontier_index, passes=choose[2],
                                             eligible=choose_gates[setting]))
              for chain in chains for setting in range(len(taus))]
    scored = [(c, t, m) for c, t, m in scored if m["degradation_rate"] <= epsilon
              and m["savings"] is not None and m["savings"] > 0]
    scored.sort(key=lambda ctm: (ctm[2]["degradation_rate"], -ctm[2]["savings"]))
    n = len(certify[0])
    best, best_tau, best_savings, best_bound = [], None, 0.0, None
    for chain, setting, metrics in scored:
        degradations = round(chain_metrics(chain, *certify[:2], frontier_index, passes=certify[2],
                                           eligible=certify_gates[setting])["degradation_rate"] * n)
        bound = clopper_pearson_upper(degradations, n, delta)
        if bound > epsilon:
            break  # fixed sequence: stop at the first failure
        if metrics["savings"] > best_savings:
            best, best_tau, best_savings, best_bound = chain, taus[setting], metrics["savings"], bound
    return best, best_tau, best_bound


def checks_matrix(checks: pd.DataFrame, tasks: pd.DataFrame, models) -> np.ndarray:
    """Measured check results (columns task_id, model, passed) as a tasks x models
    matrix. A missing or unknown result counts as a pass: no check, nothing caught."""
    return prepare_checks(checks, tasks, models)[0]


def run_cascade_experiment(frame, frontier: str, output, *, epsilons=(0.02, 0.01, 0.005),
                           catches=(1.0, 0.95, 0.9), checks: pd.DataFrame | None = None,
                           seeds=range(10), fractions=(0.4, 0.3, 0.3), delta: float = 0.05,
                           max_length: int = 3, top_k: int = 8, gate_method: str | None = None,
                           gate_thresholds=DEFAULT_GATE_THRESHOLDS,
                           gate_fractions=(0.35, 0.1, 0.2, 0.2, 0.15),
                           catch_rates: dict | None = None, gate_embeddings=None) -> dict:
    """For each seed, split tasks into choose / certify / test parts. Pick a chain
    two ways: "uncertified" (lowest cost with degradation <= epsilon on the choose
    part) and "certified" (also passes a Clopper-Pearson bound <= epsilon on the
    certify part, see ``certify_chain``). Score both on the test part.

    Checks are simulated from assumed catch rates, or taken from measured
    ``checks`` (task_id, model, passed), reported as catch "measured".
    ``catch_rates`` (model name -> rate, every non-frontier model) adds a simulated
    setting reported as catch "per-model": each model's wrong answers are caught at
    its own rate, independently of the task.
    Writes cascade-runs.csv, cascade-summary.csv, cascade-summary.json, report.md.

    With ``gate_method`` (a router predictor), each seed instead splits tasks by
    ``gate_fractions`` into train / score-calibration / choose / certify / test
    parts, fits the predictor on the first two, and adds a "gated_certified" pick:
    the certified search over every chain paired with every ``gate_thresholds``
    value or no gate (see ``certify_gated_chain``). ``gate_embeddings`` is an optional
    NPZ of fixed prompt embeddings for the predictor (as for ``tine run``); the
    default is TF-IDF fitted on the train part.
    """
    if gate_method is not None and gate_method not in METHODS:
        raise ValueError(f"Unknown gate method: {gate_method!r}; choose one of {METHODS}")
    gated = gate_method is not None
    parts = 5 if gated else 3
    fractions = tuple(_probability(value, "fraction") for value in (gate_fractions if gated else fractions))
    if len(fractions) != parts or not np.isclose(sum(fractions), 1, rtol=0, atol=1e-12) or min(fractions) <= 0:
        raise ValueError(f"fractions must be {parts} positive shares summing to 1")
    gate_thresholds = tuple(_probability(value, "gate threshold") for value in gate_thresholds)
    if gated and not gate_thresholds:
        raise ValueError("A gated experiment requires at least one gate threshold")
    if gate_embeddings is not None and not gated:
        raise ValueError("gate_embeddings requires a gate method")
    epsilons = tuple(_probability(value, "epsilon") for value in epsilons)
    catches = tuple(_probability(value, "catch") for value in catches)
    seeds = [_integer(value, "seed") for value in seeds]
    delta = _probability(delta, "delta", open_interval=True)
    max_length = _integer(max_length, "max_length", minimum=1)
    top_k = _integer(top_k, "top_k", minimum=1)
    if not seeds or not epsilons or (checks is None and not catches and not catch_rates):
        raise ValueError("Require nonempty seeds, epsilons, and simulated catch settings")
    if catch_rates is not None and checks is not None:
        raise ValueError("Use measured checks or simulated per-model catch rates, not both")
    synthetic = bool(any(
        column in frame and frame[column].eq("synthetic").any()
        for column in ("cost_source", "success_label_source")
    ))
    frame, tasks, excluded = complete_tasks(frame, frontier)
    if tasks.tenant.nunique() != 1:
        raise ValueError("Run the cascade experiment on one tenant at a time")
    models = sorted(frame.model.unique())
    frontier_index = models.index(frontier)
    outcomes = matrix(frame, tasks, models, "success", boolean=True)
    costs = matrix(frame, tasks, models, "cost_usd")
    measured, coverage = (None, None) if checks is None else prepare_checks(checks, tasks, models)
    lookup = load_embeddings(gate_embeddings, tasks) if gate_embeddings is not None else None
    model_catches = None
    if catch_rates is not None:
        missing = sorted(set(models) - {frontier} - set(catch_rates))
        if missing:
            raise ValueError(f"catch_rates lack models: {missing}")
        # The frontier's answer is final, so its check is never consulted.
        model_catches = [_probability(catch_rates.get(model, 1.0), "catch rate") for model in models]
    seeds = list(seeds)
    runs = []
    for seed in seeds:
        part = np.random.default_rng(seed).choice(parts, size=len(tasks), p=list(fractions))
        if len(set(part)) < parts:
            raise ValueError("Too few tasks for the experiment's separate parts")
        choose, certify, test = (part == parts - 3 + offset for offset in range(3))
        scores = None
        if gated:
            # complete_tasks left one task per prompt, so the parts share no prompt.
            def part_rows(index):
                return frame.loc[frame.task_id.isin(tasks.task_id[part == index])]
            router = Router(frontier, seed=seed, embedding_lookup=lookup).fit(part_rows(0), part_rows(1))
            if router.model_names != models:
                raise ValueError("Gate predictor models do not match the outcome models")
            scores = router.predict_scores(tasks[TASK_COLUMNS], method=gate_method)
        chains = candidate_chains(outcomes[choose], costs[choose], frontier_index, max_length, top_k)
        settings = [("measured", measured)] if measured is not None else [
            (catch, simulated_passes(outcomes, catch, np.random.default_rng(10_000 + seed)))
            for catch in catches]
        if model_catches is not None:
            settings.append(("per-model", simulated_passes(
                outcomes, model_catches, np.random.default_rng(10_000 + seed))))
        for catch, passes in settings:
            def rows(mask):
                return (outcomes[mask], costs[mask], passes[mask],
                        None if scores is None else scores[mask])
            for epsilon in epsilons:
                chain, bound = certify_chain(chains, rows(choose), rows(certify), frontier_index,
                                             epsilon=epsilon, delta=delta)
                picks = {
                    "uncertified": (select_chain(*rows(choose)[:2], frontier_index, epsilon=epsilon,
                                                 passes=rows(choose)[2], chains=chains), None, None),
                    "certified": (chain, None, bound),
                }
                if gated:
                    picks["gated_certified"] = certify_gated_chain(
                        chains, (None, *gate_thresholds), rows(choose), rows(certify),
                        frontier_index, epsilon=epsilon, delta=delta)
                for method, (chain, tau, bound) in picks.items():
                    test_outcomes, test_costs, test_passes, test_scores = rows(test)
                    metrics = chain_metrics(chain, test_outcomes, test_costs, frontier_index,
                                            passes=test_passes, eligible=gate(test_scores, tau))
                    runs.append({"seed": seed, "catch": catch, "epsilon": epsilon, "method": method,
                                 "chain": " > ".join([models[j] for j in chain] + [frontier]),
                                 "gate_tau": np.nan if tau is None else tau,
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
                    gated_runs=("gate_tau", "count"), gate_tau_median=("gate_tau", "median"),
                    frontier_call_rate=("frontier_call_rate", "mean"),
                    mean_calls=("mean_calls", "mean"),
                    most_common_chain=("chain", lambda c: c.value_counts().index[0]))
               .reset_index())
    result = {"frontier": frontier, "models": models, "complete_tasks": len(tasks),
              "synthetic": synthetic,
              "excluded_tasks": len(excluded), "seeds": len(seeds), "fractions": list(fractions),
              "delta": delta, "check_mode": "measured" if measured is not None else "simulated",
              "check_coverage": coverage,
              "gate": ({"method": gate_method, "thresholds": list(gate_thresholds),
                        "features": "supplied_embeddings" if lookup else "train_only_tfidf",
                        "embedding_sha256": embedding_fingerprint(lookup, tasks) if lookup else None,
                        "scope": "Predictor fitted per seed on the train and score-calibration parts only."}
                       if gated else None),
              "cost_scope": "Recorded model-generation costs only; check execution costs are excluded.",
              "certification_scope": "Per seed/epsilon/check setting under IID assumptions; not simultaneous across this sweep. Simulated checks certify only the assumed check process.",
              "summary": summary.astype(object).where(summary.notna(), None).to_dict(orient="records")}
    table = summary.round(4)
    report = "\n".join([
        "# Tine escalation-chain experiment", "",
        "Contains SYNTHETIC outcomes — not benchmark evidence." if synthetic else "Existing graded outcomes; held-out test split.", "",
        f"Frontier: `{frontier}`. Complete tasks: {len(tasks)}. Seeds: {len(seeds)}.",
        f"Each seed splits tasks {fractions} into "
        f"{'train / score-calibration / ' if gated else ''}choose / certify / test parts.",
        "`uncertified` picks the cheapest chain whose degradation on the choose part is <= epsilon.",
        f"`certified` also requires a one-sided Clopper-Pearson bound <= epsilon (delta={delta})",
        "on the certify part, testing chains in a fixed order and stopping at the first failure.",
        *([f"`gated_certified` runs the certified search over each chain with no gate or with a `{gate_method}`",
           f"predictor gate at one of {gate_thresholds}: a cheaper model is tried only where its predicted",
           "success is at least that threshold. `gated_runs` counts runs whose pick uses a gate."]
          if gated else []),
        "",
        "`catch` is the share of wrong answers the check catches: simulated for numbers and",
        "for `per-model` (supplied rate per model, applied independently of the task),",
        "observed for `measured`. Benchmark labels come from hidden tests, so catch=1.0 is an",
        "optimistic upper bound. Degradation = frontier solved the task but the chain's final",
        "answer did not. Savings are against always using the frontier, on recorded costs.", "",
        result["cost_scope"], result["certification_scope"], "",
        (f"Measured check coverage: {coverage['known_checks']}/{coverage['expected_checks']}. "
         f"Unknown checks: {coverage['unknown_checks']}; these pass without evidence. "
         "Check records are matched by identity; the caller must verify they refer to the same recorded answers."
         if coverage is not None else "Check outcomes are simulated; these are not measured checker-performance results."), "",
        table.to_markdown(index=False) if _has_tabulate() else table.to_string(index=False),
        "",
    ])
    with experiment_output(output) as staged:
        runs.to_csv(staged / "cascade-runs.csv", index=False)
        summary.to_csv(staged / "cascade-summary.csv", index=False)
        (staged / "cascade-summary.json").write_text(
            json.dumps(result, indent=2, default=float, allow_nan=False) + "\n", encoding="utf-8")
        (staged / "report.md").write_text(report, encoding="utf-8")
    return result


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True
