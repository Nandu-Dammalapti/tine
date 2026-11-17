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
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

# The offline toolkit (numpy, pandas, scikit-learn, matplotlib) is imported inside
# ``freeze_policy``, its only user here. The proxy imports this module to run a policy
# and must start without loading any of it.

POLICY_VERSION = 1
BUILT_IN_CHECKS = ("agreement", "none")
# No single call or check costs this much. A larger figure is a mistake in whatever
# reported it, and adding it up would only hide the real total.
MAX_COST_USD = 1_000_000.0
# What recorded answers hold where a model gave none (``scripts/agreement_pairs.py``).
NO_ANSWER_WORDS = ("None", "[None]", "null")
# Where ``freeze_policy`` writes the digest of everything else in the file.
DIGEST_FIELD = "content_sha256"
FREEZE_UNCHANGED = "matches the digest written when it was frozen"
FREEZE_EDITED = "edited after freezing"
FREEZE_NO_DIGEST = "no digest (frozen before digests)"
# How agreement compares answers (``canonical_answer``). A policy without the field
# compares them exactly, as every policy frozen before the field existed did.
ANSWER_COMPARE_MODES = ("exact", "math")


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
                  single_check: str = "none", shortlist: str = "savings", tenant: str | None = None,
                  answer_pattern: str | None = None, answer_compare: str | None = None) -> dict:
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

    ``tenant`` names the tenant the policy is for when it differs from the tenant in
    the outcomes; the outcomes' tenant is then kept in provenance. ``answer_pattern``
    is the regular expression the proxy reads a comparable answer with (the last
    match, its first group if it has one). It must read the same answer the recorded
    answers behind ``checks`` held, or agreement live is not the agreement measured.
    ``answer_compare`` is how two answers are compared (``canonical_answer``); it must
    be the comparison ``checks`` were measured with (``scripts/agreement_pairs.py
    --answer-compare``). None writes nothing, which means ``exact``.

    When ``checks`` also say whether each candidate's answer equals the frontier's
    (the ``matches_frontier`` column ``scripts/agreement_pairs.py`` writes), the
    policy records how often the chain's accepted answer differed from the frontier's
    on the graded tasks without being a loss (``drift_baseline``, see
    ``_drift_baseline``). Shadow mode's drift check allows for that share on top of
    ``epsilon``. Checks without the column give a policy without the block.

    The digest of the policy's content is written to ``provenance`` last, so a file
    edited afterwards can be told apart (``freeze_state``). It is not a signature:
    anyone who edits the file can write a new one.

    ``single_check="none"`` means a single model's answer is served as it is. That is
    refused when the chosen chain holds a single model whose recorded check rejected
    at least one answer: the chain was then certified with a check, and a policy that
    names none would serve every answer of that model unchecked under a guarantee
    that does not cover it.
    """
    import numpy as np

    from tine.calibration import _probability
    from tine.cascade import candidate_chains, certify_chain, chain_metrics
    from tine.checks import check_matches, prepare_checks
    from tine.pipeline import complete_tasks, matrix

    if not isinstance(single_check, str) or not single_check or single_check == "agreement":
        raise ValueError("single_check must be a check name other than 'agreement'")
    if tenant is not None and (not isinstance(tenant, str) or not tenant.strip()):
        raise ValueError("tenant must be a nonempty name")
    if answer_pattern is not None:
        check_answer_pattern(answer_pattern)
    if answer_compare is not None and answer_compare not in ANSWER_COMPARE_MODES:
        raise ValueError(f"answer_compare must be one of {list(ANSWER_COMPARE_MODES)}")
    epsilon = _probability(epsilon, "epsilon", open_interval=True)
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
    if single_check == "none":
        checked = sorted(models[index] for index in chain
                         if "+" not in models[index] and not passes[:, index].all())
        if checked:
            raise ValueError(
                f"The chain was certified with a check that rejected answers from {checked}, but "
                "single_check is 'none', which would serve them unchecked. Name the check those results "
                "measured (--single-check, for example tests+judge:sonnet)")
    on_certify = chain_metrics(chain, *rows(certify)[:2], frontier_index, passes=passes[certify])
    on_test = chain_metrics(chain, *rows(test)[:2], frontier_index, passes=passes[test])
    stages = [_stage(models[index], single_check) for index in chain]
    baseline = _drift_baseline(chain, stages, outcomes, passes, check_matches(checks, tasks, models), frontier_index,
                               {"choose": choose, "certify": certify, "test": test}, answer_compare or "exact")
    identity = hashlib.sha256()
    for task_id in sorted(tasks.task_id):
        identity.update(str(task_id).encode("utf-8") + b"\n")
    source = str(tasks.tenant.iloc[0])
    policy = {
        "version": POLICY_VERSION, "tenant": tenant if tenant is not None else source,
        "task_type": str(tasks.task_type.iloc[0]),
        "frontier": frontier, "stages": stages,
        **({"answer_pattern": answer_pattern} if answer_pattern is not None else {}),
        **({"answer_compare": answer_compare} if answer_compare is not None else {}),
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
        **({"drift_baseline": baseline} if baseline is not None else {}),
        "provenance": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed,
            "fractions": list(fractions), "complete_tasks": len(tasks), "excluded_tasks": len(excluded),
            "candidates": len(models) - 1, "task_ids_sha256": identity.hexdigest(), "check_coverage": coverage,
            **({"outcomes_tenant": source} if tenant is not None and tenant != source else {})},
    }
    policy["provenance"][DIGEST_FIELD] = policy_digest(policy)
    return policy


def _drift_baseline(chain, stages, outcomes, passes, matches, frontier_index: int, parts: dict,
                    answer_compare: str) -> dict | None:
    """How often the chain's accepted answer differed from the frontier's on the graded
    tasks, and how often that was not a loss: what shadow mode's drift check allows
    for on top of the quality limit (``tine.shadow.drift``).

    The proxy cannot see right or wrong live, only whether an accepted answer equals
    the frontier's. A differing answer is not always a loss: the frontier can be the
    one that is wrong, both can be wrong in different ways, and the same value can be
    written in two ways. ``matches`` says, per task and candidate, whether the
    candidate's answer equals the frontier's (``tine.checks.check_matches``). The
    chain is walked per task as it runs: the first stage whose check passed accepts.
    An answer accepted by a named check is not compared, as in shadow mode; its task
    stays in ``tasks``, because the quality limit is a share of all tasks. A differing
    answer is a loss when the frontier is right and the accepted candidate is wrong.

    The counts the drift check uses are those of the certify and test parts together.
    The choose part is where the chain was picked, so it is kept apart in
    ``by_part``, with the other two, for whoever wants to look again.

    None, which writes no block, when the chain is empty, the checks do not say
    whether answers match, no stage is compared, or the match of a compared accepted
    answer is unknown.
    """
    if not chain or matches is None or all(stage["check"] not in BUILT_IN_CHECKS for stage in stages):
        return None
    import numpy as np  # only freeze_policy calls this; see the note at the top of the module

    outcomes = np.asarray(outcomes, dtype=bool)
    reach = np.ones(len(outcomes), dtype=bool)
    accepted, compared, differing, lost = (np.zeros(len(outcomes), dtype=bool) for _ in range(4))
    for index, stage in zip(chain, stages):
        took = reach & passes[:, index]
        reach = reach & ~passes[:, index]
        accepted |= took
        if stage["check"] not in BUILT_IN_CHECKS:
            continue
        if any(value is None for value in matches[took, index]):
            return None
        differs = took & np.array([value is False for value in matches[:, index]], dtype=bool)
        compared |= took
        differing |= differs
        lost |= differs & outcomes[:, frontier_index] & ~outcomes[:, index]

    def counts(mask) -> dict:
        return {"tasks": int(mask.sum()), "compared": int((mask & compared).sum()),
                "differing": int((mask & differing).sum()),
                "differing_not_lost": int((mask & differing & ~lost).sum())}

    graded = parts["certify"] | parts["test"]
    whole = counts(graded)
    if not whole["tasks"]:
        return None
    return {"parts": ["certify", "test"], "answer_compare": answer_compare,
            "tasks": whole["tasks"], "accepted": int((graded & accepted).sum()), "compared": whole["compared"],
            "differing": whole["differing"], "differing_lost": int((graded & differing & lost).sum()),
            "differing_not_lost": whole["differing_not_lost"],
            "by_part": {name: counts(mask) for name, mask in parts.items()}}


# More graded tasks than any policy was frozen from: a count above this in a policy
# file's drift baseline is a mistake, and the bound worked out from it would fail.
DRIFT_BASELINE_MAX_TASKS = 10**9


def check_answer_pattern(pattern) -> None:
    """Refuse an answer pattern the proxy could not use."""
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("answer_pattern must be a nonempty regular expression")
    try:
        re.compile(pattern)
    except re.error:
        raise ValueError("answer_pattern must be a valid regular expression") from None


def policy_digest(policy: dict) -> str:
    """SHA-256 of the policy's canonical JSON (sorted keys, no spaces), leaving out the
    digest field itself. Numbers that are not plain JSON are written as floats, as
    ``dumps`` writes them, so the digest of a policy and of its file agree."""
    content = dict(policy)
    if isinstance(content.get("provenance"), dict):
        content["provenance"] = {key: value for key, value in content["provenance"].items() if key != DIGEST_FIELD}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_state(policy: dict) -> str:
    """Whether a policy is still what ``freeze_policy`` wrote: ``FREEZE_UNCHANGED``,
    ``FREEZE_EDITED``, or ``FREEZE_NO_DIGEST`` for a file without a digest (policies
    frozen before digests were written, and hand-written ones). A matching digest shows
    only that the file was not edited carelessly; it is not a signature."""
    provenance = policy.get("provenance") if isinstance(policy, dict) else None
    if not isinstance(provenance, dict) or DIGEST_FIELD not in provenance:
        return FREEZE_NO_DIGEST
    try:
        digest = policy_digest(policy)
    except (TypeError, ValueError, RecursionError):  # content no JSON file can hold
        return FREEZE_EDITED
    return FREEZE_UNCHANGED if provenance[DIGEST_FIELD] == digest else FREEZE_EDITED


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
    guarantee = policy.get("guarantee")
    if stages and not (isinstance(guarantee, dict) and guarantee.get("certified") is True):
        raise ValueError("A policy with stages must carry a certified guarantee")
    if isinstance(guarantee, dict):
        # A limit, confidence or bound written as a string, or out of range, would slip
        # past the comparison below and still read as certified.
        for name in ("epsilon", "delta"):
            if guarantee.get(name) is not None and not _fraction(guarantee[name], open_interval=True):
                raise ValueError(f"A policy's {name} must be a number strictly between 0 and 1")
        if guarantee.get("bound") is not None and not _fraction(guarantee["bound"]):
            raise ValueError("A policy's bound must be a number between 0 and 1")
        if guarantee.get("certified") is True:
            missing = [name for name in ("epsilon", "delta", "bound") if guarantee.get(name) is None]
            if missing:
                raise ValueError(f"A certified policy must state its {', '.join(missing)}")
            if guarantee["bound"] > guarantee["epsilon"]:
                raise ValueError("A certified policy's bound cannot be above its epsilon")
    if policy.get("answer_pattern") is not None:
        try:
            check_answer_pattern(policy["answer_pattern"])
        except ValueError:
            raise ValueError("A policy's answer_pattern must be a valid, nonempty regular expression") from None
    if policy.get("answer_compare") is not None and policy["answer_compare"] not in ANSWER_COMPARE_MODES:
        raise ValueError(f"A policy's answer_compare must be one of {list(ANSWER_COMPARE_MODES)}")
    if "drift_baseline" in policy:
        # The drift check turns these counts into the share of differing answers it
        # allows for; counts that cannot be ones would give a level that means nothing.
        baseline = policy["drift_baseline"] if isinstance(policy["drift_baseline"], dict) else {}
        harmless, differing, compared, tasks = (baseline.get(name) for name in (
            "differing_not_lost", "differing", "compared", "tasks"))
        if (not all(type(value) is int for value in (harmless, differing, compared, tasks))
                or not 0 <= harmless <= differing <= compared <= tasks or not 1 <= tasks <= DRIFT_BASELINE_MAX_TASKS):
            raise ValueError("A policy's drift_baseline must hold whole numbers, with tasks at least 1 and at most "
                             f"{DRIFT_BASELINE_MAX_TASKS:,}, and 0 <= differing_not_lost <= differing <= compared "
                             "<= tasks")
    return policy


def answer_mode(policy: dict) -> str:
    """How ``policy``'s agreement compares answers: its ``answer_compare``, else exact."""
    return policy.get("answer_compare") or "exact"


def _fraction(value, *, open_interval: bool = False) -> bool:
    """Whether ``value`` is a number in [0, 1], or in (0, 1) with ``open_interval``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0 < value < 1 if open_interval else 0 <= value <= 1


# What ``canonical_answer``'s math mode drops, in order: notation that does not
# change an answer's value.
_TEXT = re.compile(r"\\text\{([^{}]*)\}")
_MATH_DROPS = (
    re.compile(r"\\(?:left|right)(?![A-Za-z])"),  # \left( and \right), not \rightarrow
    re.compile(r"\^\{\\circ\}|\^\\circ"),  # degree signs
    re.compile(r"\\[!,; ]"),  # thin, medium and thick spaces, and a backslash space
    re.compile(r"\\%|%"),  # percent signs, escaped or not
    re.compile(r"\\\$"),  # escaped dollar signs
)
_DFRAC = re.compile(r"\\[dt]frac(?![A-Za-z])")
_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+")


def canonical_answer(answer, mode: str = "exact") -> str | None:
    r"""The form of ``answer`` that agreement compares, or None for no answer.

    ``exact`` collapses runs of whitespace to one space and changes nothing else.

    ``math`` is for answers written as LaTeX, such as the content of ``\boxed{...}``.
    It unwraps ``\text{...}``; drops ``\left`` and ``\right``, the spaces ``\!``,
    ``\,``, ``\;`` and ``\ ``, degree signs (``^\circ``, ``^{\circ}``), percent signs
    (``\%``, ``%``) and escaped dollar signs (``\$``); writes ``\dfrac`` and
    ``\tfrac`` as ``\frac``; and removes all whitespace, ``$`` at either end and a
    trailing ``.``. If what is left is a plain decimal number (commas allowed between
    groups of three digits), it is written in one exact form: ``14.50``, ``14.5\%``
    and ``$14.5$`` all become ``14.5``, and ``1,234`` becomes ``1234``. Numbers are
    compared exactly, never within a tolerance: agreement means the same value.

    Math mode does not catch the same value in two notations (``0.5`` and
    ``\frac{1}{2}``, ``\frac12`` and ``\frac{1}{2}``, ``9.9\%`` and ``0.099``,
    ``1e3`` and ``1000``), units written differently or on one side only (``5`` and
    ``5\text{ cm}``), reordered terms (``x+1`` and ``1+x``), or equivalent
    expressions in general. Because it drops percent signs, ``50\%`` and ``50`` count
    as the same answer, and because it removes spaces, the list ``1, 234`` reads as
    the number ``1234``.

    In either mode the words recorded answers use for "no answer" count as missing,
    as they did when agreement was measured.
    """
    if mode not in ANSWER_COMPARE_MODES:
        raise ValueError(f"answer comparison must be one of {list(ANSWER_COMPARE_MODES)}")
    if not isinstance(answer, str):
        return None
    text = " ".join(answer.split())
    if not text or text in NO_ANSWER_WORDS:
        return None
    if mode == "exact":
        return text
    unwrapped = None
    while unwrapped != text:  # innermost first, so nested \text{} unwraps too
        unwrapped, text = text, _TEXT.sub(r"\1", text)
    for pattern in _MATH_DROPS:
        text = pattern.sub("", text)
    text = _DFRAC.sub(r"\\frac", text)
    text = "".join(text.split()).strip("$")
    if text.endswith("."):
        text = text[:-1].strip("$")
    if _PLAIN_NUMBER.fullmatch(text):
        try:
            value = Decimal(text.replace(",", ""))
        except InvalidOperation:  # the pattern admits only plain numbers
            value = None
        if value is not None:
            text = "0" if value == 0 else format(value.normalize(), "f")
    return text if text and text not in NO_ANSWER_WORDS else None


def same_answer(answers, mode: str = "exact") -> bool:
    """True when every answer is present and all have the same ``canonical_answer``
    in ``mode``. The default, exact, compares after collapsing whitespace only."""
    cleaned = {canonical_answer(answer, mode) for answer in answers}
    return len(cleaned) == 1 and None not in cleaned


def _agreement(policy: dict, agree):
    """The agreement test a run of ``policy`` uses: ``agree`` if the caller gave one,
    else answers compared the way the policy names (``answer_mode``)."""
    if agree is not None:
        return agree
    mode = answer_mode(policy)
    return lambda answers: same_answer(answers, mode)


def run_policy(policy: dict, call, *, agree=None, checks: dict | None = None) -> dict:
    """Run one request through ``policy``.

    ``call(model)`` performs the request on that model and returns a mapping with
    ``answer`` (the comparable final answer, or None if there is none) and
    ``cost_usd``; it may carry anything else, such as the full response. Agreement
    compares answers as the policy's ``answer_compare`` says, unless ``agree`` is
    given. A cheaper
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
    agree = _agreement(policy, agree)
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
                               **describe_failure(error)})
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


def _status(value) -> dict:
    """``{"status": code}`` for an HTTP status code, else nothing."""
    return {"status": value} if type(value) is int and 100 <= value <= 599 else {}


def describe_failure(error: Exception) -> dict:
    """What went wrong, for the record: the kind of error and, when the error carries
    an HTTP reply (a provider or webhook answered with an error status), its status."""
    try:
        status = error.response.status_code
    except Exception:  # no reply attached, or not an HTTP error at all
        status = None
    return {"error": type(error).__name__, **_status(status)}


def _failed_call(model: str, error: Exception) -> dict:
    """The record of a call that gave no usable result. A failure that knows what the
    call cost (a paid reply with nothing in it) carries it as ``cost_usd``."""
    return {"model": model, "cost_usd": known_cost(getattr(error, "cost_usd", None)), **describe_failure(error)}


def _check_run(name: str, model: str, verdict: dict) -> dict:
    """The record of one check. A check of several steps may know only part of its
    cost; it then reports that part as ``known_cost_usd`` and it is kept beside the
    unknown total, so spending that is known is not lost with the rest. A check that
    failed closed may say why (``error``, and ``status`` for an HTTP reply). A check
    that says why it did not pass (a short ``reason`` such as ``request_too_long``)
    keeps that reason."""
    run = {"check": name, "model": model, "passed": verdict.get("passed") is True,
           "cost_usd": known_cost(verdict.get("cost_usd")),
           "error": verdict["error"] if isinstance(verdict.get("error"), str) and verdict["error"] else None,
           **_status(verdict.get("status"))}
    partial = known_cost(verdict.get("known_cost_usd"))
    if run["cost_usd"] is None and partial is not None:
        run["known_cost_usd"] = partial
    if not run["passed"] and isinstance(verdict.get("reason"), str) and verdict["reason"]:
        run["reason"] = verdict["reason"]
    if isinstance(verdict.get("steps"), list):
        run["steps"] = [_step(step) for step in verdict["steps"] if isinstance(step, dict)]
    return run


def _step(step: dict) -> dict:
    """The record of one step of a named check (``tine.runtime_checks``): its type,
    whether it passed, its cost and why it did not pass or failed, if it said."""
    return {"type": step["type"] if isinstance(step.get("type"), str) else None,
            "passed": step.get("passed") is True, "cost_usd": known_cost(step.get("cost_usd")),
            **{key: step[key] for key in ("reason", "error") if isinstance(step.get(key), str) and step[key]},
            **_status(step.get("status"))}


async def run_stages_async(policy: dict, call, *, agree=None, checks: dict | None = None,
                           deadline_seconds: float | None = None) -> dict:
    """The cheaper stages of ``policy`` for one request, for an asynchronous caller.

    Like ``run_policy`` without its last step: the frontier is not called, because
    the caller either has its answer already (shadow mode) or calls it itself when
    nothing is accepted (the proxy). ``call(model)`` is awaited; the members of a
    stage are called together. A named check may be a plain or an awaitable function.

    Returns ``accepted``, and when accepted the ``stage``, the model that
    ``served_by`` it and its ``result``; always the ``calls`` and ``checks`` made, so
    that every cent spent on the way is accounted for. Each check names the
    ``checked_model`` whose answer it read (None for a skipped stage) and, for a
    named check that lists them, its ``steps``.

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
    agree = _agreement(policy, agree)
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
                               **describe_failure(error), "checked_model": model})
            return False
        checking.remove((name, model))
        check_runs.append({**_check_run(name, model, verdict), "checked_model": model})
        return check_runs[-1]["passed"]

    async def stages():
        for index, stage in enumerate(policy["stages"]):
            skip = getattr(checks.get(stage["check"]), "skip_reason", None)
            if skip:
                # No model answered, so no answer was checked.
                check_runs.append({**_check_run(stage["check"], stage["models"][0],
                                                {"passed": False, "cost_usd": 0.0, "reason": skip}),
                                   "checked_model": None})
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
                           "error": "DeadlineExceeded", "checked_model": model} for name, model in checking)
    return accepted or {"accepted": False, "stage": None, "served_by": None, "result": None, "check": None,
                        "calls": calls, "checks": check_runs}


def dumps(policy: dict) -> str:
    return json.dumps(validate_policy(policy), indent=2, allow_nan=False, default=float) + "\n"
