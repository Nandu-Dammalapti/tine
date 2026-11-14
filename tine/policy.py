"""A frozen Gacrux policy for one tenant and task type, and the code that runs it.

The experiments in ``tine.cascade`` repeat a choice over many seeds to measure it.
A deployment needs the choice made once and written down: which cheaper models to
try, in which order, what check accepts their answer, and the evidence that the
quality limit holds. ``freeze_policy`` makes that choice from recorded outcomes and
measured checks, exactly as one seed of the certified experiment does, and returns
it as plain data. ``run_policy`` executes such a policy on a live request: try a
stage, accept its answer if its check passes, otherwise move on, and end at the
frontier model.

Routing has a cost and the customer bears it (AGENTS.md section 2). ``run_policy``
therefore returns the cost of the answer that was returned and, separately, the cost
of every other call it made to get there: second opinions and attempts that were
escalated. Nothing here calls a provider itself; the caller supplies ``call``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from tine.calibration import _probability
from tine.cascade import candidate_chains, certify_chain, chain_metrics
from tine.checks import prepare_checks
from tine.pipeline import complete_tasks, matrix

POLICY_VERSION = 1
CHECKS = ("agreement", "none")


def _stage(name: str) -> dict:
    """A candidate named ``A+B`` is a group whose members must agree; a single model
    has no check of its own and is accepted as it is."""
    members = name.split("+")
    return {"models": members, "check": "agreement" if len(members) > 1 else "none"}


def freeze_policy(frame, frontier: str, checks, *, epsilon: float, delta: float = 0.05, seed: int = 42,
                  fractions=(0.4, 0.3, 0.3), max_length: int = 3, top_k: int = 8) -> dict:
    """Choose and certify one chain, and describe it as data.

    Tasks are split by ``seed`` into choose / certify / test parts. The chain is
    picked and certified as in ``run_cascade_experiment`` (fixed-sequence
    Clopper-Pearson test, see ``certify_chain``); the test part gives an estimate of
    savings and accuracy on tasks that played no part in the choice. If nothing
    certifies, the policy has no stages and every request goes to the frontier.

    ``checks`` holds measured check results (task_id, model, passed) for the
    candidates in ``frame``, as written by ``scripts/agreement_pairs.py``.
    """
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta", open_interval=True)
    frame, tasks, excluded = complete_tasks(frame, frontier)
    if tasks.tenant.nunique() != 1 or tasks.task_type.nunique() != 1:
        raise ValueError("Freeze a policy for one tenant and one task type at a time")
    models = sorted(frame.model.unique())
    frontier_index = models.index(frontier)
    outcomes = matrix(frame, tasks, models, "success", boolean=True)
    costs = matrix(frame, tasks, models, "cost_usd")
    passes, coverage = prepare_checks(checks, tasks, models)
    part = np.random.default_rng(seed).choice(3, size=len(tasks), p=list(fractions))
    if len(set(part)) < 3:
        raise ValueError("Too few tasks for separate choose, certify and test parts")
    choose, certify, test = (part == index for index in range(3))

    def rows(mask):
        return outcomes[mask], costs[mask], passes[mask]

    chains = candidate_chains(outcomes[choose], costs[choose], frontier_index, max_length, top_k)
    chain, bound = certify_chain(chains, rows(choose), rows(certify), frontier_index, epsilon=epsilon, delta=delta)
    on_certify = chain_metrics(chain, *rows(certify)[:2], frontier_index, passes=passes[certify])
    on_test = chain_metrics(chain, *rows(test)[:2], frontier_index, passes=passes[test])
    identity = hashlib.sha256()
    for task_id in sorted(tasks.task_id):
        identity.update(str(task_id).encode("utf-8") + b"\n")
    return {
        "version": POLICY_VERSION, "tenant": str(tasks.tenant.iloc[0]), "task_type": str(tasks.task_type.iloc[0]),
        "frontier": frontier, "stages": [_stage(models[index]) for index in chain],
        "guarantee": {
            "epsilon": epsilon, "delta": delta, "certified": bool(chain),
            "statement": ("With confidence 1 - delta, on tasks drawn like the certify tasks, the share of tasks "
                          "the frontier solves and this policy does not is at most epsilon."
                          if chain else "Nothing certified at this limit: every request goes to the frontier."),
            "bound": bound, "certify_tasks": int(certify.sum()),
            "certify_degradation_rate": on_certify["degradation_rate"] if chain else 0.0},
        "estimate": {
            "scope": "held-out test tasks, recorded outcomes and costs; includes the cost of every call",
            "test_tasks": int(test.sum()), "savings": on_test["savings"], "accuracy": on_test["accuracy"],
            "frontier_accuracy": on_test["frontier_accuracy"], "degradation_rate": on_test["degradation_rate"],
            "frontier_call_rate": on_test["frontier_call_rate"]},
        "provenance": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed,
            "fractions": list(fractions), "complete_tasks": len(tasks), "excluded_tasks": len(excluded),
            "candidates": len(models) - 1, "task_ids_sha256": identity.hexdigest(), "check_coverage": coverage},
    }


def validate_policy(policy: dict) -> dict:
    """Reject a policy this code cannot run faithfully. Returns it unchanged."""
    if not isinstance(policy, dict) or policy.get("version") != POLICY_VERSION:
        raise ValueError(f"Policy version must be {POLICY_VERSION}")
    if not isinstance(policy.get("frontier"), str) or not policy["frontier"]:
        raise ValueError("Policy needs a frontier model")
    stages = policy.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Policy stages must be a list")
    for stage in stages:
        members = stage.get("models") if isinstance(stage, dict) else None
        if not isinstance(members, list) or not members or not all(isinstance(m, str) and m for m in members):
            raise ValueError("Each stage needs a nonempty list of model names")
        if stage.get("check") not in CHECKS:
            raise ValueError(f"Unknown check {stage.get('check')!r}; this version runs {CHECKS}")
        if stage["check"] == "agreement" and len(set(members)) < 2:
            raise ValueError("An agreement check needs at least two different models")
        if policy["frontier"] in members:
            raise ValueError("The frontier is the final fallback and cannot be a stage member")
    if stages and not (isinstance(policy.get("guarantee"), dict) and policy["guarantee"].get("certified") is True):
        raise ValueError("A policy with stages must carry a certified guarantee")
    return policy


def same_answer(answers) -> bool:
    """True when every answer is present and equal after collapsing whitespace."""
    cleaned = {" ".join(answer.split()) if isinstance(answer, str) else None for answer in answers}
    return len(cleaned) == 1 and None not in cleaned and "" not in cleaned


def run_policy(policy: dict, call, *, agree=same_answer) -> dict:
    """Run one request through ``policy``.

    ``call(model)`` performs the request on that model and returns a mapping with
    ``answer`` (the comparable final answer, or None if there is none) and
    ``cost_usd``; it may carry anything else, such as the full response. A cheaper
    model that raises is treated as having no answer and the stage is abandoned. A
    frontier failure is raised to the caller: there is nothing left to fall back to.

    Returns the accepted call's mapping under ``result``, the model that served it,
    the stage it came from (``"frontier"`` for the fallback), ``answer_cost_usd`` for
    that call and ``routing_cost_usd`` for every other call made, with the calls listed.
    A model that appears in two stages is called once per request; the frozen
    estimate charges it each time, so live cost can only be lower on that account.
    """
    validate_policy(policy)
    calls, seen = [], {}

    def attempt(model: str):
        if model in seen:  # asked in an earlier stage of this request: reuse, do not pay twice
            return seen[model]
        try:
            result = call(model)
        except Exception as error:  # a cheaper model's failure must not fail the request
            calls.append({"model": model, "cost_usd": None, "error": type(error).__name__})
            seen[model] = None
            return None
        calls.append({"model": model, "cost_usd": result.get("cost_usd"), "error": None})
        seen[model] = result
        return result

    def finish(result, model: str, stage):
        known = [entry["cost_usd"] for entry in calls if entry["cost_usd"] is not None]
        answer_cost = result.get("cost_usd")
        total = sum(known)
        return {"result": result, "served_by": model, "stage": stage, "calls": calls,
                "answer_cost_usd": answer_cost,
                "routing_cost_usd": total - answer_cost if answer_cost is not None else total,
                "unknown_cost_calls": sum(entry["cost_usd"] is None for entry in calls)}

    for index, stage in enumerate(policy["stages"]):
        results = []
        for model in stage["models"]:
            result = attempt(model)
            if result is None:
                break
            results.append(result)
        if len(results) < len(stage["models"]):
            continue
        if stage["check"] == "none" or agree([result.get("answer") for result in results]):
            return finish(results[0], stage["models"][0], index)
    result = call(policy["frontier"])
    calls.append({"model": policy["frontier"], "cost_usd": result.get("cost_usd"), "error": None})
    return finish(result, policy["frontier"], "frontier")


def dumps(policy: dict) -> str:
    return json.dumps(validate_policy(policy), indent=2, allow_nan=False, default=float) + "\n"
