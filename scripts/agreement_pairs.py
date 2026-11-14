#!/usr/bin/env python3
"""Build "ask two cheaper models and accept when they agree" candidates from
recorded answers, for tasks whose answers can be compared exactly.

For multiple-choice and exact-answer tasks an agreement check needs no tests and no
judge: two cheaper models answer, and their answer is accepted only if it is the
same. This script turns canonical outcomes plus the benchmark's recorded answers
into inputs for ``tine cascade --checks``:

- an outcomes file with one row per task for every pair ``A+B`` and for the
  frontier. A pair costs both models' recorded cost, because both are always called;
- a checks file saying, per task and pair, whether the two answers agree.

The cascade then treats a pair as one candidate whose check is agreement, so chains,
certification and savings all include the cost of the second opinion. No model is
called. Answers are compared after trimming and collapsing whitespace; an empty or
missing answer never agrees.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.data import read_outcomes, write_outcomes  # noqa: E402


def normalize(answer) -> str | None:
    if answer is None:
        return None
    if not isinstance(answer, str):
        answer = json.dumps(answer, sort_keys=True)
    text = " ".join(answer.split())
    return text if text and text not in ("None", "[None]", "null") else None


def recorded_answers(bench_dir: Path) -> dict[tuple[str, str], str | None]:
    """(task_id, model) -> normalised recorded answer, from each model's latest run."""
    answers = {}
    for model_dir in sorted(path for path in Path(bench_dir).iterdir() if path.is_dir()):
        runs = sorted(model_dir.glob("*.json"))
        if not runs:
            continue
        data = json.loads(runs[-1].read_text(encoding="utf-8"))
        for record in data["records"]:
            task_id = f"{data['dataset_name']}:{data['split']}:{record['index']}"
            answers[(task_id, data["model_name"])] = normalize(record.get("prediction"))
    return answers


def answers_from_csv(path: Path) -> dict[tuple[str, str], str | None]:
    """(task_id, model) -> normalised answer, from a CSV with those three columns."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return {(row["task_id"], row["model"]): normalize(row["answer"]) for row in csv.DictReader(stream)}


def build_pairs(outcomes, answers: dict, frontier: str, *, max_cost_share: float = 0.5,
                models: list[str] | None = None, group_size: int = 2, include_singles: bool = False):
    """Returns (group outcomes frame, check rows, report). A group of ``group_size``
    models (2 = a pair) is kept if its mean cost is at most ``max_cost_share`` of the
    frontier's mean cost. A group agrees only when every member gives the same answer."""
    if frontier not in set(outcomes.model):
        raise ValueError(f"Frontier {frontier!r} is absent from the outcomes")
    if not 0 < max_cost_share <= 1:
        raise ValueError("max_cost_share must be in (0, 1]")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be an integer of at least 2")
    mean_cost = outcomes.groupby("model").cost_usd.mean()
    candidates = sorted(model for model in mean_cost.index
                        if model != frontier and (models is None or model in models))
    pairs = [group for group in combinations(candidates, group_size)
             if sum(mean_cost[model] for model in group) <= max_cost_share * mean_cost[frontier]]
    if not pairs:
        raise ValueError("No pair of models is cheap enough; raise max_cost_share or pass models")
    by_model = {model: rows.set_index("task_id") for model, rows in outcomes.groupby("model")}
    frames, checks, agreement = [by_model[frontier].reset_index()], [], {}
    if include_singles:
        # Each cheaper model alone with a check that always passes: "just use the cheaper
        # model", the alternative an agreement check has to beat.
        for model in candidates:
            if mean_cost[model] <= max_cost_share * mean_cost[frontier]:
                frames.append(by_model[model].reset_index())
                checks.extend({"task_id": task_id, "model": model, "passed": True}
                              for task_id in by_model[model].index)
    for group in pairs:
        members = [by_model[model] for model in group]
        shared = members[0].index
        for member in members[1:]:
            shared = shared.intersection(member.index)
        members = [member.loc[shared] for member in members]
        name = "+".join(group)
        pair = members[0].reset_index().copy()
        pair["model"] = name
        pair["cost_usd"] = sum(member.cost_usd.to_numpy() for member in members)
        for column in ("input_tokens", "output_tokens"):
            if column in pair:
                pair[column] = sum(member[column].to_numpy() for member in members)
        for column in ("cached_tokens", "latency_ms"):
            if column in pair:
                pair[column] = None
        agree = []
        for task_id in shared:
            given = {answers.get((task_id, model)) for model in group}
            same = len(given) == 1 and None not in given
            agree.append(same)
            checks.append({"task_id": task_id, "model": name, "passed": same})
        # When the answers agree, every member is right or every member is wrong; when
        # they differ the group is not accepted, so its recorded success is never used.
        pair["success"] = members[0].success.to_numpy()
        frames.append(pair)
        agreement[name] = round(sum(agree) / len(agree), 4) if agree else None
    import pandas as pd
    frame = pd.concat(frames, ignore_index=True)
    return frame, checks, {"frontier": frontier, "group_size": group_size, "pairs": len(pairs),
                           "models_in_pairs": candidates, "include_singles": include_singles,
                           "max_cost_share": max_cost_share, "agreement_rate": agreement}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="canonical outcomes for one task group")
    parser.add_argument("--bench-dir", type=Path,
                        help="extracted benchmark split folder holding <model>/*.json with recorded answers")
    parser.add_argument("--answers", type=Path,
                        help="CSV (task_id, model, answer), instead of --bench-dir")
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--output", type=Path, required=True, help="pair outcomes JSONL")
    parser.add_argument("--checks", type=Path, required=True, help="agreement checks CSV")
    parser.add_argument("--max-cost-share", type=float, default=0.5,
                        help="keep pairs costing at most this share of the frontier's mean cost")
    parser.add_argument("--models", nargs="*", help="limit pairs to these models")
    parser.add_argument("--group-size", type=int, default=2,
                        help="models that must all agree (2 = pairs, 3 = triples)")
    parser.add_argument("--include-singles", action="store_true",
                        help="also offer each cheaper model alone, accepted without any check")
    args = parser.parse_args(argv)
    if bool(args.bench_dir) == bool(args.answers):
        raise SystemExit("Pass exactly one of --bench-dir and --answers")
    answers = recorded_answers(args.bench_dir) if args.bench_dir else answers_from_csv(args.answers)
    frame, checks, report = build_pairs(read_outcomes(args.input), answers,
                                        args.frontier, max_cost_share=args.max_cost_share, models=args.models,
                                        group_size=args.group_size, include_singles=args.include_singles)
    write_outcomes(frame, args.output)
    args.checks.parent.mkdir(parents=True, exist_ok=True)
    with args.checks.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task_id", "model", "passed"])
        writer.writeheader()
        writer.writerows(checks)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
