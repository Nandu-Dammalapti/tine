"""How many graded tasks a chain needs before it can certify at a quality limit.

``tine.cascade.certify_chain`` passes a chain when the one-sided Clopper-Pearson
upper bound on its degradation rate, measured on a certify part of ``n`` tasks, is
at most ``epsilon``. A degradation is a task the frontier solves and the chain
misses; the rate is per task, not per frontier-solved task. That test has a simple
shape: for every ``n`` there is a largest number of degradations that still passes.
Everything here follows from that number and the binomial distribution, so the
answers are exact rather than simulated.

Three things the numbers do not cover, and the caller must keep in mind:

- Tasks are assumed independent and identically distributed, and the chain's true
  degradation rate is assumed known. It is not known; pass a range of rates.
- The probability of passing is not monotone in ``n``. One more allowed failure
  arrives in steps, and between steps more tasks only add chances to fail. So "the
  smallest n with 80% power" can be followed by larger n with less. ``stable=True``
  gives the smallest n from which the power never drops below the target again.
- ``certify_chain`` tests chains in a fixed order and stops at the first failure. A
  chain is reached only if every chain before it passed, so the single-chain
  probability is an upper limit. See ``sequence_pass_bounds``. With a few hundred
  candidate chains this is the larger effect, not a correction: the numbers apply
  as they stand only when one chain is named before the tasks are graded
  (``docs/PLANNING.md``).
"""

from __future__ import annotations

from math import ceil, floor, log, sqrt

import numpy as np
from scipy.stats import binom

from tine.calibration import (
    _integer,
    _probability,
    clopper_pearson_lower,
    clopper_pearson_upper,
)

# Searches stop here rather than run for ever when a target is almost impossible.
MAX_TASKS = 10_000_000

SPLITS = ("fixed", "random")

# Weight of a random certify size that the average may leave out on each side.
_TAIL = 1e-15

# failure_steps is pure and is asked for the same limits again and again.
_STEPS: dict[tuple[float, float], list[int | None]] = {}


def _passes(k: int, n: int, epsilon: float, delta: float) -> bool:
    """The certification test itself: is the bound for k failures in n within epsilon?"""
    return clopper_pearson_upper(k, n, delta) <= epsilon


def max_failures(n: int, epsilon: float, delta: float) -> int | None:
    """Largest number of degradations in ``n`` certify tasks that still certifies.

    Returns None when even zero degradations give a bound above ``epsilon`` (always
    the case for ``n == 0``, where the bound is 1, unless ``epsilon`` is 1). Uses
    ``clopper_pearson_upper`` directly, so it agrees with ``certify_chain``. The
    bound grows with the number of failures, which makes a binary search exact.
    """
    n = _integer(n, "n")
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta", open_interval=True)
    if not _passes(0, n, epsilon, delta):
        return None
    low, high = 0, n  # low always passes
    while low < high:
        middle = (low + high + 1) // 2
        if _passes(middle, n, epsilon, delta):
            low = middle
        else:
            high = middle - 1
    return low


def tasks_needed(epsilon: float, delta: float, failures: int = 0, *,
                 max_tasks: int = MAX_TASKS) -> int | None:
    """Smallest nonempty certify part on which ``failures`` degradations still pass.

    None when no size up to ``max_tasks`` passes (always so for ``epsilon == 0``).
    For a fixed number of failures the bound falls as ``n`` grows, so the search
    doubles ``n`` until the test passes and then bisects. With ``failures=0`` this
    is ``tine.calibration.minimum_risk_tasks`` for one comparison.
    """
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta", open_interval=True)
    failures = _integer(failures, "failures")
    max_tasks = _integer(max_tasks, "max_tasks")
    return _tasks_needed_from(max(1, failures), failures, epsilon, delta, max_tasks)


def _tasks_needed_from(start: int, failures: int, epsilon: float, delta: float,
                       max_tasks: int) -> int | None:
    """``tasks_needed`` when the answer is known to be at least ``start``."""
    if epsilon == 0 or start > max_tasks:
        return None
    if _passes(failures, start, epsilon, delta):
        return start
    low, step = start, 1  # low always fails
    while True:
        high = min(low + step, max_tasks)
        if _passes(failures, high, epsilon, delta):
            break
        if high == max_tasks:
            return None
        low, step = high, step * 2
    while high - low > 1:
        middle = (low + high) // 2
        if _passes(failures, middle, epsilon, delta):
            high = middle
        else:
            low = middle
    return high


def failure_steps(epsilon: float, delta: float, max_tasks: int) -> list[int]:
    """``steps[k]`` is ``tasks_needed`` for k failures, for every k that fits in ``max_tasks``.

    This is the whole test in one list: ``n`` certify tasks allow k failures exactly
    when ``steps[k] <= n``. It lets the functions below handle many values of ``n``
    without recomputing a bound for each.
    """
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta", open_interval=True)
    max_tasks = _integer(max_tasks, "max_tasks")
    # ``known`` ends with the first step beyond every limit asked for so far.
    known = _STEPS.setdefault((epsilon, delta), [])
    while not known or (known[-1] is not None and known[-1] <= max_tasks):
        failures = len(known)
        start = max(known[-1] if known else 1, failures, 1)  # one more failure never needs fewer tasks
        known.append(_tasks_needed_from(start, failures, epsilon, delta, MAX_TASKS * 100))
    return [step for step in known if step is not None and step <= max_tasks]


def pass_probability(n: int, epsilon: float, delta: float, true_rate: float) -> float:
    """Chance that a chain with degradation rate ``true_rate`` certifies on ``n`` tasks.

    The number of degradations is binomial(n, true_rate) for independent tasks, and
    the chain passes when it is at most ``max_failures``. When ``true_rate`` is
    ``epsilon`` or more this is at most ``delta``: that is the guarantee.
    """
    true_rate = _probability(true_rate, "true_rate")
    allowed = max_failures(n, epsilon, delta)
    if allowed is None:
        return 0.0
    return float(binom.cdf(allowed, n, true_rate))


def _power_horizon(epsilon: float, delta: float, true_rate: float, power: float) -> int:
    """A certify size from which the pass probability is certain to stay >= ``power``.

    The Clopper-Pearson bound is never looser than Hoeffding's, so k failures pass
    when k/n + sqrt(log(1/delta) / 2n) <= epsilon, and Hoeffding again bounds the
    chance of more failures than that. Solving for n gives a (generous) horizon,
    which makes the search for the last dip below ``power`` finite and exact.
    """
    gap = epsilon - true_rate
    return ceil(((sqrt(log(1 / delta) / 2) + sqrt(log(1 / (1 - power)) / 2)) / gap) ** 2) + 1


def tasks_for_power(epsilon: float, delta: float, true_rate: float, power: float = 0.8, *,
                    stable: bool = False, max_tasks: int = MAX_TASKS) -> int | None:
    """Smallest certify part on which the chain passes with probability >= ``power``.

    None when ``true_rate >= epsilon``: the test then passes with probability at
    most ``delta`` whatever the size, so it cannot be made likely. Also None when
    the answer, or with ``stable=True`` the size needed to be sure of it, would
    exceed ``max_tasks``. Rates just under ``epsilon`` need very large sizes and
    take seconds to search.

    The pass probability rises only where one more failure becomes allowed and
    falls in between, so the first size to reach ``power`` is always one of
    ``failure_steps``. With ``stable=True`` the answer is instead the smallest size
    from which every larger size also reaches ``power``; this is the number to plan
    with when the certify size is not controlled to the task.
    """
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta", open_interval=True)
    true_rate = _probability(true_rate, "true_rate")
    power = _probability(power, "power", open_interval=True)
    max_tasks = _integer(max_tasks, "max_tasks")
    if true_rate >= epsilon:
        return None
    if true_rate == 0:
        # No failure ever happens, so passing only needs the zero-failure size.
        return tasks_needed(epsilon, delta, 0, max_tasks=max_tasks)
    horizon = _power_horizon(epsilon, delta, true_rate, power)
    if not stable:
        steps = failure_steps(epsilon, delta, min(horizon, max_tasks))
        reached = np.flatnonzero(binom.cdf(np.arange(len(steps)), steps, true_rate) >= power)
        return steps[reached[0]] if len(reached) else None
    if horizon > max_tasks:
        return None
    # One step past the horizon, so the last dip before it is seen in full.
    steps = failure_steps(epsilon, delta, horizon)
    following = _tasks_needed_from(steps[-1] if steps else 1, len(steps), epsilon, delta,
                                   MAX_TASKS * 100)
    # Within a stretch the probability only falls, so its lowest point is its last
    # size, just before the next step. A stretch whose last size reaches ``power``
    # reaches it everywhere, so the answer is the step after the last dip below it.
    last = np.asarray([*steps[1:], following]) - 1
    dips = np.flatnonzero(binom.cdf(np.arange(len(steps)), last, true_rate) < power)
    if not len(dips):
        return steps[0] if steps else following
    return [*steps, following][dips[-1] + 1]


def sequence_pass_bounds(probabilities) -> tuple[float, float]:
    """Limits on the chance that every chain in a fixed-sequence prefix passes.

    ``certify_chain`` stops at the first failure, so a chain is certified only if it
    and every chain tested before it pass. Given each one's own pass probability,
    the joint chance lies between ``max(0, 1 - sum(1 - p))`` and ``min(p)`` whatever
    the dependence between chains. Chains share models and tasks, so their failures
    overlap and the truth is usually nearer the upper limit, but that is not
    measured here.
    """
    values = [_probability(value, "probability") for value in probabilities]
    if not values:
        raise ValueError("Require at least one pass probability")
    return max(0.0, 1.0 - sum(1.0 - value for value in values)), min(values)


def rate_estimate(failures: int, sample: int, confidence: float = 0.95) -> dict:
    """A degradation rate estimated from ``failures`` in ``sample`` tasks.

    Returns the point estimate with one-sided Clopper-Pearson limits, each at
    ``confidence`` (together they are a two-sided interval at 2 * confidence - 1).
    Plan with the upper limit as well as the point estimate: a count of two or
    three says little about a rate. If the chain was picked on the same tasks the
    count is optimistic, and no interval corrects for that.
    """
    sample = _integer(sample, "sample", minimum=1)
    failures = _integer(failures, "failures")
    confidence = _probability(confidence, "confidence", open_interval=True)
    if failures > sample:
        raise ValueError("failures cannot exceed sample")
    return {
        "failures": failures,
        "sample": sample,
        "rate": failures / sample,
        "lower": clopper_pearson_lower(failures, sample, 1 - confidence),
        "upper": clopper_pearson_upper(failures, sample, 1 - confidence),
        "confidence": confidence,
    }


def certify_tasks(total_tasks: int, certify_fraction: float) -> int:
    """Certify-part size for a fixed split: ``total_tasks * certify_fraction``, rounded down."""
    total_tasks = _integer(total_tasks, "total_tasks")
    certify_fraction = _probability(certify_fraction, "certify_fraction")
    # The small tolerance stops 0.3 * 10 from landing on 2.
    return min(total_tasks, floor(total_tasks * certify_fraction + 1e-9))


def split_pass_probability(total_tasks: int, certify_fraction: float, epsilon: float,
                           delta: float, true_rate: float, *, split: str = "fixed") -> float:
    """``pass_probability`` when the certify part is a share of ``total_tasks``.

    ``split="fixed"`` gives the certify part exactly ``certify_tasks``.
    ``split="random"`` is what ``run_cascade_experiment`` and ``tine freeze-policy``
    do: each task is assigned to a part independently, so the certify size is
    binomial(total_tasks, certify_fraction) and the answer is the average over that
    size (sizes in the far tails, 2e-15 of the weight at most, are left out).
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    total_tasks = _integer(total_tasks, "total_tasks")
    certify_fraction = _probability(certify_fraction, "certify_fraction")
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta", open_interval=True)
    true_rate = _probability(true_rate, "true_rate")
    if epsilon == 1:
        # Every upper bound passes, including 1.0 for an empty certify part.
        # failure_steps starts at nonempty sizes, so it omits that probability mass.
        return 1.0
    if split == "fixed":
        return pass_probability(certify_tasks(total_tasks, certify_fraction), epsilon, delta,
                                true_rate)
    return _random_split_pass(total_tasks, certify_fraction, true_rate,
                              failure_steps(epsilon, delta, total_tasks))


def _random_split_pass(total_tasks: int, certify_fraction: float, true_rate: float,
                       steps: list[int]) -> float:
    """Average pass probability over a binomial certify size, given ``failure_steps``."""
    if not steps or steps[0] > total_tasks:
        return 0.0
    # Only the far tails of the certify size are left out: by the definition of the
    # quantiles, at most _TAIL of its weight on each side.
    low = int(binom.ppf(_TAIL, total_tasks, certify_fraction))
    high = int(binom.isf(_TAIL, total_tasks, certify_fraction))
    sizes = np.arange(max(steps[0], low), min(total_tasks, high) + 1)
    if not len(sizes):
        return 0.0
    allowed = np.searchsorted(np.asarray(steps), sizes, side="right") - 1
    weights = binom.pmf(sizes, total_tasks, certify_fraction)
    return float(np.sum(weights * binom.cdf(allowed, sizes, true_rate)))


def plan(total_tasks: int, certify_fraction: float, epsilon: float, delta: float = 0.05,
         true_rate: float | None = None, *, split: str = "fixed") -> dict:
    """What ``total_tasks`` graded tasks can certify at ``epsilon`` with this split.

    ``certify_tasks`` is the planned certify size (the expected size for a random
    split, rounded down). ``max_failures`` and the bounds describe that size. With
    ``true_rate``, ``pass_probability`` is the chance of certifying one chain of
    that rate, for the fixed or random split asked for. ``zero_failure_tasks`` is
    the certify size below which nothing can certify at all.
    """
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta", open_interval=True)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    size = certify_tasks(total_tasks, certify_fraction)
    allowed = max_failures(size, epsilon, delta)
    result = {
        "total_tasks": total_tasks,
        "certify_fraction": certify_fraction,
        "split": split,
        "certify_tasks": size,
        "epsilon": epsilon,
        "delta": delta,
        "max_failures": allowed,
        "zero_failure_bound": clopper_pearson_upper(0, size, delta),
        "bound_at_max_failures": None if allowed is None else clopper_pearson_upper(allowed, size, delta),
        "zero_failure_tasks": tasks_needed(epsilon, delta, 0),
        "true_rate": true_rate,
        "pass_probability": None,
    }
    if true_rate is not None:
        result["pass_probability"] = split_pass_probability(
            total_tasks, certify_fraction, epsilon, delta, true_rate, split=split)
    return result


def total_tasks_for_power(certify_fraction: float, epsilon: float, delta: float,
                          true_rate: float, power: float = 0.8, *, split: str = "fixed",
                          stable: bool = True, max_tasks: int = MAX_TASKS) -> int | None:
    """Inverse of ``plan``: graded tasks needed in total for a pass probability >= ``power``.

    None when ``true_rate >= epsilon`` or the answer exceeds ``max_tasks``.
    With ``stable=True``, also None when the search limit prevents establishing
    that the power stays above the target. A first-crossing search does not need
    that additional evidence and includes ``max_tasks`` itself.

    For a fixed split this is the smallest total whose certify part reaches
    ``tasks_for_power`` (``stable`` is passed through; the default True means no
    larger total does worse). For a random split it is the smallest total at which
    the average over the random certify size reaches ``power``, found by checking
    every total upwards from the first that could work. Averaging smooths the steps
    but does not remove them, so with ``stable=True`` the search continues until no
    total up to the size where every likely certify part is past the last dip falls
    below ``power``.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    max_tasks = _integer(max_tasks, "max_tasks")
    certify_fraction = _probability(certify_fraction, "certify_fraction")
    if certify_fraction == 0:
        return None
    needed = tasks_for_power(epsilon, delta, true_rate, power, stable=stable, max_tasks=max_tasks)
    if needed is None:
        return None
    if split == "fixed":
        total = max(needed, floor(needed / certify_fraction) - 1)
        while certify_tasks(total, certify_fraction) < needed:
            total += 1
        while total > needed and certify_tasks(total - 1, certify_fraction) >= needed:
            total -= 1
        return total if total <= max_tasks else None
    return _random_total_for_power(certify_fraction, epsilon, delta, true_rate, power,
                                   stable=stable, max_tasks=max_tasks)


def _random_total_for_power(certify_fraction: float, epsilon: float, delta: float,
                            true_rate: float, power: float, *, stable: bool,
                            max_tasks: int) -> int | None:
    """``total_tasks_for_power`` for a random split, by exact scan over totals."""
    first = tasks_for_power(epsilon, delta, true_rate, power, max_tasks=max_tasks)
    if first is None:
        return None
    if not stable:
        # The first crossing does not require a stable-power horizon. Scan in
        # growing batches so a large limit does not force us to build millions
        # of unused failure steps before finding a small answer.
        start, end = first, min(max_tasks, first * 2)
        while True:
            steps = failure_steps(epsilon, delta, end)
            for total in range(start, end + 1):
                if _random_split_pass(total, certify_fraction, true_rate, steps) >= power:
                    return total
            if end == max_tasks:
                return None
            start, end = end + 1, min(max_tasks, end * 2)
    settled = tasks_for_power(epsilon, delta, true_rate, power, stable=True, max_tasks=max_tasks)
    if settled is None:
        return None
    # Past this total the certify part is below ``settled`` with chance under 1e-9,
    # so the average cannot dip below ``power`` again by more than that.
    end = settled
    while end <= max_tasks and binom.cdf(settled - 1, end, certify_fraction) > 1e-9:
        end = ceil(end * 1.05) + 1
    if end > max_tasks:
        return None
    steps = failure_steps(epsilon, delta, end)
    answer = None
    # No total below ``first`` can work: every certify size it allows is short of power.
    for total in range(first, end + 1):
        if _random_split_pass(total, certify_fraction, true_rate, steps) < power:
            answer = None
        elif answer is None:
            answer = total
    return answer
