#!/usr/bin/env python3
"""Import xRouteBench recorded outcomes and answers into Tine's formats.

xRouteBench (``ulab-ai/xRouteBench`` on Hugging Face, the benchmark of the
LLMRouter library) records, for every query and each of 18 open models, the model's
response, a score and token counts. This script reads the ``llmrouter_generic``
parquet files and writes:

- canonical outcomes (``--output``), one row per query and model, for the tasks whose
  score is binary. Cost is ``input_tokens * input_price + output_tokens *
  output_price`` from the benchmark's own price table, as its README defines it;
- an answers file (``--answers``): the final answer extracted from each response, for
  ``scripts/agreement_pairs.py --answers``. Multiple choice: the last parenthesised
  letter. GSM8K-style: the last number. MATH: the last ``\\boxed{...}``. Code tasks
  have no comparable answer and get none;
- a report on stdout, including how often "extracted answer equals the reference"
  matches the benchmark's own score where that can be checked.

No model is called and no response is executed. Needs ``pyarrow`` to read parquet.
Tasks scored with a fractional metric (``squad``, F1) are left out: canonical
outcomes are success or failure, and turning a fraction into one would be inventing
a threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.data import write_outcomes  # noqa: E402

GROUPS = {
    "multiple-choice": ["arc_challenge", "boolq", "commonsense_qa", "hellaswag", "mmlu", "mmlu_pro", "openbook_qa"],
    "math": ["aime_2020_2024", "gsm8k", "math"],
    "coding": ["human_eval", "mbpp"],
}
FRACTIONAL = {"squad"}
_LETTER = re.compile(r"\(\s*([A-Za-z])\s*\)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def choice_answer(response: str) -> str | None:
    found = _LETTER.findall(response or "")
    return found[-1].upper() if found else None


def number_answer(response: str) -> str | None:
    found = _NUMBER.findall(response or "")
    if not found:
        return None
    text = found[-1].replace(",", "")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def boxed_answer(response: str) -> str | None:
    """Contents of the last ``\\boxed{...}``, braces balanced, whitespace removed."""
    start = (response or "").rfind("\\boxed{")
    if start < 0:
        return None
    depth, begin = 0, start + len("\\boxed{")
    for position in range(begin, len(response)):
        if response[position] == "{":
            depth += 1
        elif response[position] == "}":
            if depth == 0:
                text = "".join(response[begin:position].split())
                return text or None
            depth -= 1
    return None


EXTRACTORS = {"em_mc": choice_answer, "GSM8K": number_answer, "MATH": boxed_answer}


def task_key(task_name: str, query: str) -> str:
    return f"xroutebench:{task_name}:{hashlib.sha256(query.encode('utf-8')).hexdigest()[:12]}"


def convert(frame, prices, *, tasks: list[str], task_type: str):
    """Returns (canonical outcomes frame, answer rows, report)."""
    import pandas as pd

    frame = frame[frame.task_name.isin(tasks)].copy()
    if frame.empty:
        raise ValueError("No rows for the chosen tasks")
    fractional = sorted(set(frame.task_name) & FRACTIONAL)
    if fractional:
        raise ValueError(f"Tasks with fractional scores cannot be imported: {fractional}")
    frame["performance"] = frame.performance.astype(float)
    if not frame.performance.isin([0.0, 1.0]).all():
        raise ValueError("A chosen task has scores other than 0 and 1")
    frame["task_id"] = [task_key(name, query) for name, query in zip(frame.task_name, frame["query"])]
    before = len(frame)
    frame = frame.drop_duplicates(["task_id", "model_name"], keep="first")
    price = prices.set_index("model_name")
    missing = sorted(set(frame.model_name) - set(price.index))
    if missing:
        raise ValueError(f"No price for models: {missing}")
    tokens_in, tokens_out = frame.input_tokens.astype(int), frame.output_tokens.astype(int)
    cost = (tokens_in.to_numpy() * price.loc[frame.model_name, "input_price_per_1m"].to_numpy()
            + tokens_out.to_numpy() * price.loc[frame.model_name, "output_price_per_1m"].to_numpy()) / 1e6
    outcomes = pd.DataFrame({
        "tenant": "benchmark", "task_id": frame.task_id.to_numpy(), "prompt": frame["query"].to_numpy(),
        "task_type": task_type, "model": frame.model_name.to_numpy(),
        "success": frame.performance.to_numpy() == 1.0, "cost_usd": cost,
        "input_tokens": tokens_in.to_numpy(), "output_tokens": tokens_out.to_numpy(),
        "cached_tokens": None, "latency_ms": frame.response_time.astype(float).to_numpy() * 1000,
        "timestamp": None, "source_task": frame.task_name.to_numpy(), "source_split": frame.split.to_numpy()})
    answers, checked, matched = [], 0, 0
    for task_id, model, metric, response, truth, score in zip(
            frame.task_id, frame.model_name, frame.metric, frame.response, frame.ground_truth, frame.performance):
        extract = EXTRACTORS.get(metric)
        answer = extract(response) if extract else None
        answers.append({"task_id": task_id, "model": model, "answer": answer or ""})
        if metric == "em_mc":  # the reference is the letter itself, so the extraction can be audited
            checked += 1
            matched += (answer is not None and answer == str(truth).strip().upper()) == (score == 1.0)
    report = {"task_type": task_type, "tasks": sorted(set(frame.task_name)), "queries": int(frame.task_id.nunique()),
              "models": int(frame.model_name.nunique()), "rows": len(frame), "duplicate_rows_dropped": before - len(frame),
              "answers_extracted": round(sum(bool(row["answer"]) for row in answers) / len(answers), 4),
              "multiple_choice_extraction_matches_score": round(matched / checked, 4) if checked else None}
    return outcomes, answers, report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--routing", type=Path, nargs="+", required=True,
                        help="llmrouter_generic parquet files (train and test)")
    parser.add_argument("--candidates", type=Path, required=True, help="llm_candidates/train.parquet")
    parser.add_argument("--group", choices=sorted(GROUPS), help="a named set of tasks")
    parser.add_argument("--tasks", nargs="*", help="task names, instead of --group")
    parser.add_argument("--task-type", help="task_type to record (default: xroutebench-<group>)")
    parser.add_argument("--output", type=Path, required=True, help="canonical outcomes JSONL")
    parser.add_argument("--answers", type=Path, required=True, help="extracted answers CSV")
    args = parser.parse_args(argv)
    if bool(args.group) == bool(args.tasks):
        raise SystemExit("Pass exactly one of --group and --tasks")
    import pandas as pd

    frames = []
    for path in args.routing:
        part = pd.read_parquet(path)
        part["split"] = path.stem
        frames.append(part)
    tasks = GROUPS[args.group] if args.group else args.tasks
    task_type = args.task_type or f"xroutebench-{args.group or 'custom'}"
    outcomes, answers, report = convert(pd.concat(frames, ignore_index=True), pd.read_parquet(args.candidates),
                                        tasks=tasks, task_type=task_type)
    write_outcomes(outcomes, args.output)
    args.answers.parent.mkdir(parents=True, exist_ok=True)
    with args.answers.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task_id", "model", "answer"])
        writer.writeheader()
        writer.writerows(answers)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
