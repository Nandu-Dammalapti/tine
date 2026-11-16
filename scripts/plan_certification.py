#!/usr/bin/env python3
"""Print how many graded tasks a chain needs before it can certify at a quality limit.

Certification passes when the one-sided Clopper-Pearson bound on the degradation
rate, measured on the certify part, is at most epsilon (``tine.cascade.certify_chain``).
This script turns that test into three small tables, computed exactly by
``tine.planning``:

- what a certify part of a given size allows: how many failures, and the chance
  that a chain of a given true rate passes;
- how many certify tasks, and how many graded tasks in total under each split,
  give a chosen chance of passing;
- with ``--earlier-chains``, the same when chains tested earlier in the fixed
  sequence must pass too.

The true rate is not known. ``--observed K N`` estimates it from K degradations in
N tasks and plans for the estimate and for its upper confidence limit; ``--rate``
adds rates to try. Nothing is read from disk and no model is called.

Example:
    python scripts/plan_certification.py --epsilon 0.04 0.02 --observed 2 274 \\
        --total 274 --split 40/30/30 30/50/20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.calibration import _integer, _probability, clopper_pearson_upper  # noqa: E402
from tine.planning import (  # noqa: E402
    MAX_TASKS,
    SPLITS,
    certify_tasks,
    max_failures,
    pass_probability,
    rate_estimate,
    split_pass_probability,
    tasks_for_power,
    tasks_needed,
    total_tasks_for_power,
)


def parse_split(text: str) -> float:
    """Certify share from ``choose/certify/test`` (such as 40/30/30) or a plain fraction."""
    try:
        parts = [float(part) for part in text.split("/")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"split {text!r} is not numeric") from None
    if len(parts) == 1 and 0 < parts[0] <= 1:
        return parts[0]
    if len(parts) == 3 and min(parts) > 0 and abs(sum(parts) - 100) < 1e-9:
        return parts[1] / 100
    raise argparse.ArgumentTypeError(
        f"split {text!r} must be choose/certify/test shares summing to 100, or a certify fraction")


def check_arguments(args) -> None:
    """Raise ValueError for a number out of range, before any table is printed."""
    for epsilon in args.epsilon:
        _probability(epsilon, "--epsilon")
    for name in ("delta", "power", "confidence"):
        _probability(getattr(args, name), f"--{name}", open_interval=True)
    for rate in args.rate:
        _probability(rate, "--rate")
    for size in [*args.certify, *([] if args.total is None else [args.total])]:
        _integer(size, "a task count")
    if args.observed:
        rate_estimate(*args.observed, confidence=args.confidence)
    if args.earlier_chains < 0:
        raise ValueError("--earlier-chains must not be negative")


def percent(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def count(value: int | None, missing: str = "cannot") -> str:
    return missing if value is None else f"{value:,}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def scenarios(args) -> list[tuple[str, float]]:
    """(label, true rate) pairs to plan for: the estimate, its upper limit, any stated rates."""
    rates = []
    if args.observed:
        estimate = rate_estimate(*args.observed, confidence=args.confidence)
        rates.append((f"{percent(estimate['rate'])} (observed)", estimate["rate"]))
        rates.append((f"{percent(estimate['upper'])} (upper limit)", estimate["upper"]))
    rates += [(percent(rate), rate) for rate in args.rate]
    return rates


def certify_sizes(args) -> list[tuple[int, int | None, float | None]]:
    """(certify tasks, total tasks, certify fraction) for every size asked about."""
    sizes = [(size, None, None) for size in args.certify]
    if args.total is not None:
        sizes += [(certify_tasks(args.total, fraction), args.total, fraction)
                  for fraction in args.split]
    return sizes


def allowed_table(args, rates) -> str:
    """What each certify size allows, and the chance each scenario passes on it."""
    header = ["epsilon", "certify tasks", "failures allowed", "bound with no failures"]
    for label, _ in rates:
        header.append(f"passes at {label}")
        if args.total is not None:
            header.append("same, random split")
    rows = []
    for epsilon in args.epsilon:
        for size, total, fraction in certify_sizes(args):
            allowed = max_failures(size, epsilon, args.delta)
            row = [f"{epsilon:g}", f"{size:,}", count(allowed, "none"),
                   percent(clopper_pearson_upper(0, size, args.delta))]
            for _, rate in rates:
                row.append(percent(pass_probability(size, epsilon, args.delta, rate), 1))
                if args.total is not None:
                    row.append("n/a" if total is None else percent(split_pass_probability(
                        total, fraction, epsilon, args.delta, rate, split="random"), 1))
            rows.append(row)
    return table(header, rows)


def needed_table(args, rates, power: float) -> str:
    """Certify tasks and total graded tasks for a pass chance of at least ``power``."""
    header = ["epsilon", "true rate", "certify tasks, first reached", "certify tasks, stays above"]
    for text in args.split_text:
        header += [f"total at {text}, {split} split" for split in SPLITS]
    rows = []
    for epsilon in args.epsilon:
        for label, rate in [("0 (no failures ever)", 0.0), *rates]:
            row = [f"{epsilon:g}", label,
                   count(tasks_for_power(epsilon, args.delta, rate, power)),
                   count(tasks_for_power(epsilon, args.delta, rate, power, stable=True))]
            for fraction in args.split:
                row += [count(total_tasks_for_power(fraction, epsilon, args.delta, rate, power,
                                                    split=split)) for split in SPLITS]
            rows.append(row)
    return table(header, rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.04, 0.02],
                        help="Quality limits to plan for")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8,
                        help="Wanted chance that the chain certifies")
    parser.add_argument("--observed", type=int, nargs=2, metavar=("FAILURES", "TASKS"),
                        help="Degradations seen and the number of tasks they were seen in")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="One-sided confidence for the limits on the observed rate")
    parser.add_argument("--rate", type=float, nargs="*", default=[],
                        help="True degradation rates to plan for, as fractions")
    parser.add_argument("--total", type=int, help="Graded tasks in hand, divided by each --split")
    parser.add_argument("--certify", type=int, nargs="*", default=[],
                        help="Certify-part sizes to describe directly")
    parser.add_argument("--split", nargs="+", default=["40/30/30", "30/50/20"],
                        help="choose/certify/test shares, or a certify fraction")
    parser.add_argument("--earlier-chains", type=int, default=0,
                        help="Chains tested before this one in the fixed sequence; adds a worst-case table")
    args = parser.parse_args(argv)
    args.split_text = list(args.split)
    try:
        args.split = [parse_split(text) for text in args.split_text]
        check_arguments(args)
    except (argparse.ArgumentTypeError, ValueError) as error:
        parser.error(str(error))
    rates = scenarios(args)

    print(f"Certification passes when the Clopper-Pearson bound at confidence "
          f"{percent(1 - args.delta, 0)} is at most epsilon.")
    print("Rates are degradations per task. Tasks are assumed independent.\n")
    if args.observed:
        estimate = rate_estimate(*args.observed, confidence=args.confidence)
        print(f"Observed {estimate['failures']} of {estimate['sample']}: {percent(estimate['rate'])}. "
              f"One-sided {percent(args.confidence, 0)} limits: {percent(estimate['lower'])} "
              f"and {percent(estimate['upper'])}.")
        print("If the chain was chosen on these tasks the count is optimistic.\n")

    print("Smallest certify part that passes with k failures:\n")
    print(table(["epsilon", *[f"k = {k}" for k in range(5)]],
                [[f"{epsilon:g}", *[count(tasks_needed(epsilon, args.delta, k)) for k in range(5)]]
                 for epsilon in args.epsilon]))

    if certify_sizes(args):
        print("\nWhat each certify part allows:\n")
        print(allowed_table(args, rates))

    print(f"\nTasks needed for a {percent(args.power, 0)} chance to certify one chain "
          f'("cannot": the rate is at or over the limit, or the answer is over {MAX_TASKS:,} tasks):\n')
    print(needed_table(args, rates, args.power))
    print("\n'first reached' can be followed by larger sizes with a lower chance; "
          "'stays above' cannot. Totals use 'stays above'.")
    print("These are for one chain tested alone. The fixed-sequence test reaches a chain only "
          "if every chain ordered before it passed, so read them as lower limits "
          "(docs/PLANNING.md).")

    if args.earlier_chains:
        chains = args.earlier_chains + 1
        each = 1 - (1 - args.power) / chains
        print(f"\nWorst case with {args.earlier_chains} earlier chain(s) of the same rate that must "
              f"also pass: each needs a {percent(each, 1)} chance.\n")
        print(needed_table(args, rates, each))


if __name__ == "__main__":
    main()
