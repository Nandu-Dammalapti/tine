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
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.data import read_outcomes, validate_outcomes, write_outcomes  # noqa: E402


def normalize(answer) -> str | None:
    if answer is None:
        return None
    if not isinstance(answer, str):
        answer = json.dumps(answer, sort_keys=True)
    text = " ".join(answer.split())
    return text if text and text not in ("None", "[None]", "null") else None


def recorded_answers(bench_dir: Path, *, outcomes=None) -> dict[tuple[str, str], str | None]:
    """Latest recorded answers, checking source hashes when outcomes supply them."""
    expected = {}
    if outcomes is not None:
        outcomes = validate_outcomes(outcomes)
        if outcomes.tenant.nunique() != 1:
            raise ValueError("Read agreement answers for one tenant at a time")
        if "source_sha256" in outcomes:
            expected = {(row.task_id, row.model): row.source_sha256
                        for row in outcomes.loc[outcomes.source_sha256.notna()].itertuples()}
    answers = {}
    for model_dir in sorted(path for path in Path(bench_dir).iterdir() if path.is_dir()):
        runs = sorted(model_dir.glob("*.json"))
        if not runs:
            continue
        raw = runs[-1].read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        for record in data["records"]:
            task_id = f"{data['dataset_name']}:{data['split']}:{record['index']}"
            key = (task_id, data["model_name"])
            if key in answers:
                raise ValueError("Duplicate recorded answer for task_id/model; select one run explicitly")
            if key in expected and expected[key] != checksum:
                raise ValueError("Recorded answer source does not match the outcome source_sha256; use a matching benchmark run")
            answers[key] = normalize(record.get("prediction"))
    return answers


def build_pairs(outcomes, answers: dict, frontier: str, *, max_cost_share: float = 0.5,
                models: list[str] | None = None, group_size: int = 2):
    """Returns (group outcomes frame, check rows, report). A group of ``group_size``
    models (2 = a pair) is kept if its mean cost is at most ``max_cost_share`` of the
    frontier's mean cost. A group agrees only when every member gives the same answer."""
    outcomes = validate_outcomes(outcomes)
    if outcomes.tenant.nunique() != 1:
        raise ValueError("Build agreement groups for one tenant at a time")
    tenant = outcomes.tenant.iloc[0]
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
    names = ["+".join(group) for group in pairs]
    if len(set(names)) != len(names) or set(names) & set(outcomes.model):
        raise ValueError("Agreement group identifiers collide with existing models or another group")
    by_model = {model: rows.set_index("task_id") for model, rows in outcomes.groupby("model")}
    frames, checks, agreement = [by_model[frontier].reset_index()], [], {}
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
            if same:
                grades = [member.at[task_id, "success"] for member in members]
                known = {bool(value) for value in grades if not pd.isna(value)}
                if len(known) > 1:
                    raise ValueError("Conflicting grades for agreeing answers; verify the recorded run and grading provenance")
            agree.append(same)
            checks.append({"tenant": tenant, "task_id": task_id, "model": name, "passed": same})
        # When the answers agree, every member is right or every member is wrong; when
        # they differ the group is not accepted, so its recorded success is never used.
        pair["success"] = members[0].success.to_numpy()
        frames.append(pair)
        agreement[name] = round(sum(agree) / len(agree), 4) if agree else None
    frame = pd.concat(frames, ignore_index=True)
    return frame, checks, {"frontier": frontier, "group_size": group_size, "pairs": len(pairs),
                           "models_in_pairs": candidates,
                           "max_cost_share": max_cost_share, "agreement_rate": agreement}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="canonical outcomes for one task group")
    parser.add_argument("--bench-dir", type=Path, required=True,
                        help="extracted benchmark split folder holding <model>/*.json with recorded answers")
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--output", type=Path, required=True, help="pair outcomes JSONL")
    parser.add_argument("--checks", type=Path, required=True, help="agreement checks CSV")
    parser.add_argument("--max-cost-share", type=float, default=0.5,
                        help="keep pairs costing at most this share of the frontier's mean cost")
    parser.add_argument("--models", nargs="*", help="limit pairs to these models")
    parser.add_argument("--group-size", type=int, default=2,
                        help="models that must all agree (2 = pairs, 3 = triples)")
    args = parser.parse_args(argv)
    if len({path.resolve() for path in (args.input, args.output, args.checks)}) != 3:
        parser.error("Input outcomes, group outcomes, and checks must use distinct paths")
    outcomes = read_outcomes(args.input)
    frame, checks, report = build_pairs(outcomes, recorded_answers(args.bench_dir, outcomes=outcomes),
                                        args.frontier, max_cost_share=args.max_cost_share, models=args.models,
                                        group_size=args.group_size)
    write_outcomes(frame, args.output)
    args.checks.parent.mkdir(parents=True, exist_ok=True)
    with args.checks.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["tenant", "task_id", "model", "passed"])
        writer.writeheader()
        writer.writerows(checks)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
