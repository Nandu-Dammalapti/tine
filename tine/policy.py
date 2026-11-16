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
import math
from datetime import datetime, timezone

import numpy as np

from tine.calibration import _probability
from tine.cascade import candidate_chains, certify_chain, chain_metrics
from tine.checks import prepare_checks
from tine.pipeline import complete_tasks, matrix

POLICY_VERSION = 1
BUILT_IN_CHECKS = ("agreement", "none")
# No single call or check costs this much. A larger figure is a mistake in whatever
# reported it, and adding it up would only hide the real total.
MAX_COST_USD = 1_000_000.0
# What recorded answers hold where a model gave none (``scripts/agreement_pairs.py``).
NO_ANSWER_WORDS = ("None", "[None]", "null")


def known_cost(value) -> float | None:
    """``value`` as a cost in dollars, or None when it is not a usable amount.

    Costs reach this code from provider usage, a judge's reply and a customer's
    webhook. One that is missing, negative, not finite or absurdly large is unknown:
    it must not be added to a total, and it must not be read as zero.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        cost = float(value)
    except OverflowError:
        return None
    return cost if math.isfinite(cost) and 0 <= cost <= MAX_COST_USD else None


def _stage(name: str, single_check: str) -> dict:
    """A candidate named ``A+B`` is a group whose members must agree. A single model
    carries ``single_check``: the name of the check the recorded results measured for
    it, or ``none`` if it is accepted as it is."""
    members = name.split("+")
    return {"models": members, "check": "agreement" if len(members) > 1 else single_check}


def freeze_policy(frame, frontier: str, checks, *, epsilon: float, delta: float = 0.05, seed: int = 42,
                  fractions=(0.4, 0.3, 0.3), max_length: int = 3, top_k: int = 8,
                  single_check: str = "none", shortlist: str = "savings") -> dict:
    """Choose and certify one chain, and describe it as data.

    Tasks are split by ``seed`` into choose / certify / test parts. The chain is
    picked and certified as in ``run_cascade_experiment`` (fixed-sequence
    Clopper-Pearson test, see ``certify_chain``); the test part gives an estimate of
    savings and accuracy on tasks that played no part in the choice. If nothing
    certifies, the policy has no stages and every request goes to the frontier.

    ``checks`` holds measured check results (task_id, model, passed) for the
    candidates in ``frame``, as written by ``scripts/agreement_pairs.py`` or
    ``scripts/judge_patches.py checks``. ``single_check`` names what those results
    measured for a single model, for example ``tests+judge``; whoever runs the policy
    must supply a check of that name that does the same thing, or the guarantee does
    not describe what runs.

    ``shortlist`` is the candidate rule of ``candidate_chains``; ``measured`` ranks
    candidates by their check results on the choose part at this ``epsilon``.
    """
    if not isinstance(single_check, str) or not single_check or single_check == "agreement":
        raise ValueError("single_check must be a check name other than 'agreement'")
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

    chains = candidate_chains(outcomes[choose], costs[choose], frontier_index, max_length, top_k,
                              shortlist=shortlist, passes=passes[choose], epsilon=epsilon)
    chain, bound = certify_chain(chains, rows(choose), rows(certify), frontier_index, epsilon=epsilon, delta=delta)
    on_certify = chain_metrics(chain, *rows(certify)[:2], frontier_index, passes=passes[certify])
    on_test = chain_metrics(chain, *rows(test)[:2], frontier_index, passes=passes[test])
    identity = hashlib.sha256()
    for task_id in sorted(tasks.task_id):
        identity.update(str(task_id).encode("utf-8") + b"\n")
    return {
        "version": POLICY_VERSION, "tenant": str(tasks.tenant.iloc[0]), "task_type": str(tasks.task_type.iloc[0]),
        "frontier": frontier, "stages": [_stage(models[index], single_check) for index in chain],
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
    if not isinstance(policy, dict) or type(policy.get("version")) is not int or policy["version"] != POLICY_VERSION:
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
        if not isinstance(stage.get("check"), str) or not stage["check"]:
            raise ValueError("Each stage needs a check name")
        if len(set(members)) != len(members):
            raise ValueError("A stage cannot name the same model twice")
        if stage["check"] == "agreement" and len(members) < 2:
            raise ValueError("An agreement check needs at least two different models")
        if stage["check"] != "agreement" and len(members) != 1:
            raise ValueError("Only an agreement stage may have several models")
        if policy["frontier"] in members:
            raise ValueError("The frontier is the final fallback and cannot be a stage member")
    if stages and not (isinstance(policy.get("guarantee"), dict) and policy["guarantee"].get("certified") is True):
        raise ValueError("A policy with stages must carry a certified guarantee")
    if stages:
        # Checked only when the file states both: a certified chain has a bound within its limit.
        bound, epsilon = known_cost(policy["guarantee"].get("bound")), known_cost(policy["guarantee"].get("epsilon"))
        if bound is not None and epsilon is not None and bound > epsilon:
            raise ValueError("A certified policy's bound cannot be above its epsilon")
    return policy


def same_answer(answers) -> bool:
    """True when every answer is present and equal after collapsing whitespace. The
    words recorded answers use for "no answer" count as missing, as they did when
    agreement was measured."""
    cleaned = {" ".join(answer.split()) if isinstance(answer, str) else None for answer in answers}
    return len(cleaned) == 1 and not cleaned & {None, "", *NO_ANSWER_WORDS}


def run_policy(policy: dict, call, *, agree=same_answer, checks: dict | None = None) -> dict:
    """Run one request through ``policy``.

    ``call(model)`` performs the request on that model and returns a mapping with
    ``answer`` (the comparable final answer, or None if there is none) and
    ``cost_usd``; it may carry anything else, such as the full response. A cheaper
    model that raises is treated as having no answer and the stage is abandoned. A
    frontier failure is raised to the caller: there is nothing left to fall back to.

    A stage whose check is neither ``agreement`` nor ``none`` is decided by
    ``checks[name](result)``, which returns a mapping with ``passed`` and, if the
    check itself cost money (a judge call), ``cost_usd``. That cost is routing cost.
    A check that raises counts as a failure. Every check a policy names must be
    supplied, or nothing is run.

    Returns the accepted call's mapping under ``result``, the model that served it,
    the stage it came from (``"frontier"`` for the fallback), ``answer_cost_usd`` for
    that call and ``routing_cost_usd`` for every other call made, with the calls listed.
    When any call or check has no known cost (``unknown_cost_calls``), the routing
    cost is not known either and ``routing_cost_usd`` is None, never a partial sum
    that would read as the whole; the part that is known is ``known_routing_cost_usd``.
    A model that appears in two stages is called once per request; the frozen
    estimate charges it each time, so live cost can only be lower on that account.
    """
    validate_policy(policy)
    checks = checks or {}
    missing = sorted({stage["check"] for stage in policy["stages"]} - set(BUILT_IN_CHECKS) - set(checks))
    if missing:
        raise ValueError(f"No check supplied for: {missing}")
    calls, seen, check_runs = [], {}, []

    def attempt(model: str):
        if model in seen:  # asked in an earlier stage of this request: reuse, do not pay twice
            return seen[model]
        try:
            result = call(model)
        except Exception as error:  # a cheaper model's failure must not fail the request
            calls.append(_failed_call(model, error))
            seen[model] = None
            return None
        calls.append({"model": model, "cost_usd": known_cost(result.get("cost_usd")), "error": None})
        seen[model] = result
        return result

    def finish(result, model: str, stage):
        known = [entry["cost_usd"] for entry in calls + check_runs if entry["cost_usd"] is not None]
        answer_cost = known_cost(result.get("cost_usd"))
        total = math.fsum(known)
        unknown = sum(entry["cost_usd"] is None for entry in calls + check_runs)
        known_routing = total - answer_cost if answer_cost is not None else total
        return {"result": result, "served_by": model, "stage": stage, "calls": calls, "checks": check_runs,
                "answer_cost_usd": answer_cost,
                "routing_cost_usd": None if unknown else known_routing,
                "known_routing_cost_usd": known_routing, "unknown_cost_calls": unknown}

    def passes(name: str, model: str, result) -> bool:
        try:
            verdict = checks[name](result)
        except Exception as error:  # a broken check must not accept an answer or fail the request
            check_runs.append({"check": name, "model": model, "passed": False, "cost_usd": None,
                               "error": type(error).__name__})
            return False
        check_runs.append(_check_run(name, model, verdict))
        return check_runs[-1]["passed"]

    for index, stage in enumerate(policy["stages"]):
        results = []
        for model in stage["models"]:
            result = attempt(model)
            if result is None:
                break
            results.append(result)
        if len(results) < len(stage["models"]):
            continue
        if stage["check"] == "none":
            accepted = True
        elif stage["check"] == "agreement":
            accepted = agree([result.get("answer") for result in results])
        else:
            accepted = passes(stage["check"], stage["models"][0], results[0])
        if accepted:
            return finish(results[0], stage["models"][0], index)
    result = call(policy["frontier"])
    calls.append({"model": policy["frontier"], "cost_usd": known_cost(result.get("cost_usd")), "error": None})
    return finish(result, policy["frontier"], "frontier")


def _failed_call(model: str, error: Exception) -> dict:
    """The record of a call that gave no usable result. A failure that knows what the
    call cost (a paid reply with nothing in it) carries it as ``cost_usd``."""
    return {"model": model, "cost_usd": known_cost(getattr(error, "cost_usd", None)), "error": type(error).__name__}


def _check_run(name: str, model: str, verdict: dict) -> dict:
    """The record of one check. A check of several steps may know only part of its
    cost; it then reports that part as ``known_cost_usd`` and it is kept beside the
    unknown total, so spending that is known is not lost with the rest. A check that
    says why it did not pass (a short ``reason`` such as ``request_too_long``) keeps
    that reason."""
    run = {"check": name, "model": model, "passed": verdict.get("passed") is True,
           "cost_usd": known_cost(verdict.get("cost_usd")), "error": None}
    partial = known_cost(verdict.get("known_cost_usd"))
    if run["cost_usd"] is None and partial is not None:
        run["known_cost_usd"] = partial
    if not run["passed"] and isinstance(verdict.get("reason"), str) and verdict["reason"]:
        run["reason"] = verdict["reason"]
    return run


async def run_stages_async(policy: dict, call, *, agree=same_answer, checks: dict | None = None,
                           deadline_seconds: float | None = None) -> dict:
    """The cheaper stages of ``policy`` for one request, for an asynchronous caller.

    Like ``run_policy`` without its last step: the frontier is not called, because
    the caller either has its answer already (shadow mode) or calls it itself when
    nothing is accepted (the proxy). ``call(model)`` is awaited; the members of a
    stage are called together. A named check may be a plain or an awaitable function.

    Returns ``accepted``, and when accepted the ``stage``, the model that
    ``served_by`` it and its ``result``; always the ``calls`` and ``checks`` made, so
    that every cent spent on the way is accounted for.

    With ``deadline_seconds`` the stages together get that long. When it runs out
    nothing is accepted, so a stalled model or check delays the caller by the deadline
    and not by every stage's own timeout in turn. A call or check that was cut off is
    listed with an unknown cost: the provider may still charge for it.

    A named check may carry a ``skip_reason``: why, from the request alone, it cannot
    pass whatever a model answers (``tine.runtime_checks``). Its stage is then skipped
    before any of its models is called, and the check is listed as not passed, at no
    cost, with that reason.
    """
    import asyncio
    import inspect

    validate_policy(policy)
    checks = checks or {}
    missing = sorted({stage["check"] for stage in policy["stages"]} - set(BUILT_IN_CHECKS) - set(checks))
    if missing:
        raise ValueError(f"No check supplied for: {missing}")
    calls, seen, check_runs = [], {}, []
    calling, checking = [], []  # what is in flight, so a deadline can account for it

    async def attempt(model: str):
        if model in seen:
            return seen[model]
        calling.append(model)
        try:
            result = await call(model)
        except Exception as error:
            calling.remove(model)
            calls.append(_failed_call(model, error))
            seen[model] = None
            return None
        calling.remove(model)
        calls.append({"model": model, "cost_usd": known_cost(result.get("cost_usd")), "error": None})
        seen[model] = result
        return result

    async def passes(name: str, model: str, result) -> bool:
        checking.append((name, model))
        try:
            verdict = checks[name](result)
            if inspect.isawaitable(verdict):
                verdict = await verdict
        except Exception as error:
            checking.remove((name, model))
            check_runs.append({"check": name, "model": model, "passed": False, "cost_usd": None,
                               "error": type(error).__name__})
            return False
        checking.remove((name, model))
        check_runs.append(_check_run(name, model, verdict))
        return check_runs[-1]["passed"]

    async def stages():
        for index, stage in enumerate(policy["stages"]):
            skip = getattr(checks.get(stage["check"]), "skip_reason", None)
            if skip:
                check_runs.append(_check_run(stage["check"], stage["models"][0],
                                             {"passed": False, "cost_usd": 0.0, "reason": skip}))
                continue
            results = await asyncio.gather(*(attempt(model) for model in stage["models"]))
            if any(result is None for result in results):
                continue
            if stage["check"] == "none":
                accepted = True
            elif stage["check"] == "agreement":
                accepted = agree([result.get("answer") for result in results])
            else:
                accepted = await passes(stage["check"], stage["models"][0], results[0])
            if accepted:
                return {"accepted": True, "stage": index, "served_by": stage["models"][0], "result": results[0],
                        "check": stage["check"], "calls": calls, "checks": check_runs}
        return None

    try:
        accepted = await asyncio.wait_for(stages(), deadline_seconds)
    except TimeoutError:
        accepted = None
        calls.extend({"model": model, "cost_usd": None, "error": "DeadlineExceeded"} for model in calling)
        check_runs.extend({"check": name, "model": model, "passed": False, "cost_usd": None,
                           "error": "DeadlineExceeded"} for name, model in checking)
    return accepted or {"accepted": False, "stage": None, "served_by": None, "result": None, "check": None,
                        "calls": calls, "checks": check_runs}


def dumps(policy: dict) -> str:
    return json.dumps(validate_policy(policy), indent=2, allow_nan=False, default=float) + "\n"
