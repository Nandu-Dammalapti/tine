"""Build 4% and 2% frozen policies from recorded outcomes, offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agreement_pairs import answers_from_csv, build_pairs  # noqa: E402
from tine.artifacts import atomic_write_text, experiment_output  # noqa: E402
from tine.checks import read_checks  # noqa: E402
from tine.data import read_outcomes  # noqa: E402
from tine.policy import ANSWER_COMPARE_MODES, DIGEST_FIELD, check_answer_pattern, dumps, freeze_policy  # noqa: E402


POLICY_FILES = ((0.04, "policy-eps04.json"), (0.02, "policy-eps02.json"))
SUMMARY_FILE = "summary.json"
BUILD_FILES = frozenset([SUMMARY_FILE, *(name for _, name in POLICY_FILES)])


def build(args) -> dict:
    outcomes = read_outcomes(args.input)
    agreement = None
    if args.answers is not None:
        if outcomes.tenant.nunique() != 1:
            raise ValueError("Build agreement groups for one tenant at a time")
        answers = answers_from_csv(args.answers, tenant=outcomes.tenant.iloc[0])
        outcomes, rows, agreement = build_pairs(
            outcomes, answers, args.frontier, group_size=args.group_size,
            max_cost_share=args.max_cost_share, include_singles=args.include_singles,
            grade_conflicts=args.grade_conflicts, answer_compare=args.answer_compare or "exact",
        )
        checks = pd.DataFrame(rows)
    else:
        checks = read_checks(args.checks)
        if args.answer_pattern is None and outcomes.model.str.contains("+", regex=False).any():
            raise ValueError("Agreement candidates require --answer-pattern")

    policies = []
    for epsilon, name in POLICY_FILES:
        policy = freeze_policy(
            outcomes, args.frontier, checks, epsilon=epsilon, seed=args.seed,
            fractions=tuple(args.fractions), tenant=args.tenant, single_check=args.single_check,
            answer_pattern=args.answer_pattern, answer_compare=args.answer_compare,
        )
        coverage = policy["provenance"]["check_coverage"]["per_model"]
        for stage in policy["stages"]:
            if stage["check"] != "none":
                candidate = "+".join(stage["models"])
                measured = coverage[candidate]
                if measured["known"] != measured["expected"]:
                    raise ValueError(f"Selected check for {candidate} is missing results on complete tasks")
        policies.append((name, policy))

    first = policies[0][1]
    summary = {
        "scope": "Offline build using recorded outcomes and costs",
        **{key: first[key] for key in ("tenant", "task_type", "frontier")},
        "seed": args.seed,
        "fractions": list(args.fractions),
        **({"agreement": agreement} if agreement is not None else {}),
        "policies": [
            {
                "file": name,
                **{key: policy[key] for key in ("guarantee", "estimate", "stages")},
                "content_sha256": policy["provenance"][DIGEST_FIELD],
                "held_out_exceeds_limit": policy["estimate"]["degradation_rate"] > policy["guarantee"]["epsilon"],
            }
            for name, policy in policies
        ],
    }
    with experiment_output(args.output_dir, artifacts=BUILD_FILES) as staged:
        for name, policy in policies:
            atomic_write_text(staged / name, dumps(policy))
        atomic_write_text(staged / SUMMARY_FILE, json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Recorded outcomes CSV or JSONL")
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--answers", type=Path, help="Extracted answers CSV for building agreement candidates")
    evidence.add_argument("--checks", type=Path, help="Measured checks CSV for prepared candidate outcomes")
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tenant", help="Tenant to name in the policy, if different from the outcomes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs=3, metavar=("CHOOSE", "CERTIFY", "TEST"),
                        default=(0.4, 0.3, 0.3))
    parser.add_argument("--answer-pattern", help="Same regular expression used to extract recorded answers")
    parser.add_argument("--answer-compare", choices=ANSWER_COMPARE_MODES)
    parser.add_argument("--single-check", default="none", help="Runtime check name for prepared single-model stages")
    parser.add_argument("--include-singles", action="store_true", help="Also try unchecked single models with --answers")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-cost-share", type=float, default=0.5)
    parser.add_argument("--grade-conflicts", choices=("error", "wrong"), default="error")
    args = parser.parse_args(argv)
    if args.answers is not None:
        if args.answer_pattern is None:
            parser.error("--answers requires --answer-pattern")
        if args.single_check != "none":
            parser.error("A named --single-check requires prepared evidence via --checks")
    elif (args.include_singles or args.group_size != 2 or args.max_cost_share != 0.5
          or args.grade_conflicts != "error"):
        parser.error("Agreement build options require --answers")
    for path in (args.input, args.answers or args.checks):
        if path.resolve() in {args.output_dir.resolve() / name for name in BUILD_FILES}:
            parser.error("A build output would overwrite an input file")
    try:
        if args.answer_pattern is not None:
            check_answer_pattern(args.answer_pattern)
        summary = build(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
