#!/usr/bin/env python3
"""Have a judge model review recorded SWE-bench patches, and score the judge.

This is Gacrux stage 4 applied after the answer: a judge reads the issue and a
cheaper model's patch and says whether the patch resolves the issue. Nothing is
executed. Three steps, each a subcommand:

``prepare``  Pick the patches that already passed the free checks (static checks and
             the repository's existing tests) and write them as judge cases, with the
             grades kept in a separate key file the judge never sees.
``judge``    Send each case, one per request, to any OpenAI-compatible chat endpoint
             (a customer-hosted open model behind vLLM, Ollama or LM Studio, or a
             provider API) and record the verdict, the tokens used and the time taken.
``score``    Compare one or more verdict files with the key: how many wrong patches
             the judge rejects, how many right ones it rejects, and what it costs.

The judge's own tokens are routing cost (AGENTS.md section 2): ``score`` reports them
so they can be netted from savings. ``judge`` calls a model; it needs the task's
authorization when that model is paid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

PROMPT_VERSION = "2026-11-11"
INSTRUCTIONS = """You are reviewing a proposed code change before it is accepted.

Below are a GitHub issue from the {repository} repository and a patch that claims to resolve it. The patch already applies cleanly and does not break the project's existing tests. Decide whether it correctly and completely resolves the issue, so that tests written specifically for this issue would pass. You cannot run code; reason from the issue and the patch.

Reply with one JSON object and nothing else:
{{"verdict": "accept" or "reject", "probability": a number from 0 to 1 that the patch correctly resolves the issue, "reason": "at most 20 words"}}"""
CASE_TEMPLATE = "<issue>\n{issue}\n</issue>\n\n<patch>\n{patch}\n</patch>"
MAX_ISSUE_CHARS = 12000
MAX_PATCH_CHARS = 20000
VERDICT_FIELDS = ["case_id", "judge", "verdict", "probability", "reason", "prompt_tokens",
                  "completion_tokens", "tokens_estimated", "seconds", "prompt_version"]


def case_text(case: dict) -> str:
    return CASE_TEMPLATE.format(issue=case["issue"][:MAX_ISSUE_CHARS], patch=case["patch"][:MAX_PATCH_CHARS])


def build_messages(case: dict) -> list[dict]:
    return [{"role": "system", "content": INSTRUCTIONS.format(repository=case["repository"])},
            {"role": "user", "content": case_text(case)}]


def parse_verdict(text: str) -> dict:
    """The judge's JSON object from its reply. Raises ValueError if there is none."""
    for candidate in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        verdict = str(value.get("verdict", "")).strip().lower()
        if verdict not in ("accept", "reject"):
            continue
        try:
            probability = float(value.get("probability"))
        except (TypeError, ValueError):
            continue
        if not 0 <= probability <= 1:
            continue
        return {"verdict": verdict, "probability": probability,
                "reason": str(value.get("reason", ""))[:200]}
    raise ValueError("No verdict object with verdict and probability in the reply")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def prepare(bench_dir: Path, static_checks: Path, repo_tests: Path, dataset: Path, cases_output: Path,
            key_output: Path, *, models: set[str], sample: int | None, seed: int) -> dict:
    """Judge cases for patches that passed the static checks and the existing tests.
    Identical patches for one task become one case. The key keeps the grades apart."""
    instances = {row["instance_id"]: row for row in read_jsonl(dataset)}
    static_pass = {(row["instance_id"], row["model"]) for row in read_csv(static_checks)
                   if row["passed"].strip().lower() == "true"}
    runs = {(row["instance_id"], row["model"]): row for row in read_csv(repo_tests)}
    valid = {iid for (iid, model), row in runs.items()
             if model == "__gold__" and row["status"] == "ok" and row["resolved"] == "True"}
    grouped: dict[tuple[str, str], dict] = {}
    for model_dir in sorted(path for path in Path(bench_dir).iterdir() if path.is_dir()):
        if models and model_dir.name not in models:
            continue
        files = sorted(model_dir.glob("*.json"))
        if not files:
            continue
        data = json.loads(files[-1].read_text(encoding="utf-8"))
        for record in data["records"]:
            key = (record.get("instance_id"), data["model_name"])
            run = runs.get(key)
            if (key[0] not in valid or key not in static_pass or run is None or run["status"] != "ok"
                    or run["p2p_failed"] != "0" or not isinstance(record.get("prediction"), str)):
                continue
            entry = grouped.setdefault((key[0], run["patch_sha"]), {
                "instance_id": key[0], "patch_sha": run["patch_sha"], "patch": record["prediction"],
                "task_id": f"swe-bench:{data['split']}:{record['index']}", "models": [],
                "benchmark_right": float(record["score"]) == 1.0, "run_resolved": run["resolved"] == "True"})
            entry["models"].append(data["model_name"])
    entries = sorted(grouped.values(), key=lambda entry: (entry["instance_id"], entry["patch_sha"]))
    if sample is not None and sample < len(entries):
        entries = sorted(random.Random(seed).sample(entries, sample),
                         key=lambda entry: (entry["instance_id"], entry["patch_sha"]))
    cases_output, key_output = Path(cases_output), Path(key_output)
    cases_output.parent.mkdir(parents=True, exist_ok=True)
    key_output.parent.mkdir(parents=True, exist_ok=True)
    with cases_output.open("w", encoding="utf-8") as cases, key_output.open("w", newline="", encoding="utf-8") as key:
        writer = csv.writer(key)
        writer.writerow(["case_id", "task_id", "instance_id", "patch_sha", "models", "benchmark_right", "run_resolved"])
        for entry in entries:
            case_id = hashlib.sha256(f"{seed}|{entry['instance_id']}|{entry['patch_sha']}".encode()).hexdigest()[:12]
            instance = instances[entry["instance_id"]]
            cases.write(json.dumps({"case_id": case_id, "repository": instance["repo"],
                                    "issue": instance["problem_statement"], "patch": entry["patch"]},
                                   ensure_ascii=False) + "\n")
            writer.writerow([case_id, entry["task_id"], entry["instance_id"], entry["patch_sha"],
                             ";".join(sorted(entry["models"])), entry["benchmark_right"], entry["run_resolved"]])
    return {"cases": len(entries), "benchmark_right": sum(entry["benchmark_right"] for entry in entries),
            "tasks": len({entry["instance_id"] for entry in entries})}


def append_verdict(output: Path, row: dict) -> None:
    output = Path(output)
    new = not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=VERDICT_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in VERDICT_FIELDS})


def judge(cases_path: Path, output: Path, *, base_url: str, model: str, judge_name: str | None = None,
          api_key: str | None = None, max_tokens: int = 400, timeout: float = 300, client=None) -> dict:
    """One request per case to an OpenAI-compatible endpoint. Resumes from ``output``."""
    import httpx

    judge_name = judge_name or model
    done = {row["case_id"] for row in read_csv(output) if row["judge"] == judge_name} if Path(output).exists() else set()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = client or httpx.Client(timeout=timeout)
    counts = {"judged": 0, "skipped": len(done), "unparsed": 0}
    for case in read_jsonl(cases_path):
        if case["case_id"] in done:
            continue
        started = time.time()
        response = client.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json={
            "model": model, "messages": build_messages(case), "temperature": 0, "max_tokens": max_tokens})
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        row = {"case_id": case["case_id"], "judge": judge_name, "prompt_version": PROMPT_VERSION,
               "prompt_tokens": usage.get("prompt_tokens", ""), "completion_tokens": usage.get("completion_tokens", ""),
               "tokens_estimated": False, "seconds": round(time.time() - started, 2)}
        try:
            row.update(parse_verdict(payload["choices"][0]["message"]["content"] or ""))
        except (ValueError, KeyError, IndexError, TypeError):
            # An unusable reply is recorded as such; scoring treats it as a rejection.
            row.update({"verdict": "unparsed", "probability": "", "reason": ""})
            counts["unparsed"] += 1
        append_verdict(output, row)
        counts["judged"] += 1
    return counts


def score(key_path: Path, verdict_paths: list[Path], *, grade: str = "benchmark_right",
          thresholds=(None, 0.5, 0.7, 0.9)) -> list[dict]:
    """Per judge and threshold: wrong patches rejected, right patches rejected, the
    share of accepted patches that are right, and tokens per case. ``threshold=None``
    uses the judge's own accept/reject; a number accepts when probability >= it.
    An unparsed reply counts as a rejection: the chain would escalate."""
    key = {row["case_id"]: row[grade] == "True" for row in read_csv(key_path)}
    by_judge: dict[str, dict[str, dict]] = {}
    for path in verdict_paths:
        for row in read_csv(path):
            if row["case_id"] in key:
                by_judge.setdefault(row["judge"], {})[row["case_id"]] = row
    results = []
    for judge_name, verdicts in sorted(by_judge.items()):
        def tokens(field):
            values = [float(row[field]) for row in verdicts.values() if row[field] not in ("", None)]
            return round(sum(values) / len(values), 1) if values else None

        for threshold in thresholds:
            counts = {"right": 0, "wrong": 0, "right_rejected": 0, "wrong_rejected": 0}
            for case_id, row in verdicts.items():
                if threshold is None:
                    accepted = row["verdict"] == "accept"
                else:
                    accepted = row["probability"] not in ("", None) and float(row["probability"]) >= threshold
                right = key[case_id]
                counts["right" if right else "wrong"] += 1
                if not accepted:
                    counts["right_rejected" if right else "wrong_rejected"] += 1
            accepted_right = counts["right"] - counts["right_rejected"]
            accepted_total = accepted_right + counts["wrong"] - counts["wrong_rejected"]
            results.append({
                "judge": judge_name, "threshold": "verdict" if threshold is None else threshold,
                "cases": len(verdicts), **counts,
                "catch_rate": round(counts["wrong_rejected"] / counts["wrong"], 4) if counts["wrong"] else None,
                "false_reject_rate": round(counts["right_rejected"] / counts["right"], 4) if counts["right"] else None,
                "accepted_share": round(accepted_total / len(verdicts), 4),
                "accuracy_when_accepted": round(accepted_right / accepted_total, 4) if accepted_total else None,
                "unparsed": sum(row["verdict"] == "unparsed" for row in verdicts.values()),
                "mean_prompt_tokens": tokens("prompt_tokens"), "mean_completion_tokens": tokens("completion_tokens"),
                "tokens_estimated": any(row["tokens_estimated"] == "True" for row in verdicts.values()),
                "mean_seconds": tokens("seconds")})
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare", help="write judge cases and a separate answer key")
    make.add_argument("--bench-dir", type=Path, required=True)
    make.add_argument("--static-checks", type=Path, required=True)
    make.add_argument("--repo-tests", type=Path, required=True)
    make.add_argument("--dataset", type=Path, required=True, help="SWE-bench Verified rows as JSON lines")
    make.add_argument("--cases", type=Path, required=True)
    make.add_argument("--key", type=Path, required=True)
    make.add_argument("--models", nargs="*", default=[])
    make.add_argument("--sample", type=int)
    make.add_argument("--seed", type=int, default=42)
    run = commands.add_parser("judge", help="ask a model for verdicts (calls the model)")
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:11434/v1")
    run.add_argument("--model", required=True)
    run.add_argument("--judge-name", help="label for this judge in the output (default: the model name)")
    run.add_argument("--api-key-env", help="environment variable holding the API key, if the endpoint needs one")
    run.add_argument("--max-tokens", type=int, default=400)
    check = commands.add_parser("score", help="compare verdict files with the key")
    check.add_argument("--key", type=Path, required=True)
    check.add_argument("--verdicts", type=Path, nargs="+", required=True)
    check.add_argument("--grade", choices=("benchmark_right", "run_resolved"), default="benchmark_right")
    check.add_argument("--output", type=Path, help="also write the table as CSV")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.bench_dir, args.static_checks, args.repo_tests, args.dataset, args.cases,
                                 args.key, models=set(args.models), sample=args.sample, seed=args.seed)))
    elif args.command == "judge":
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        if args.api_key_env and not api_key:
            raise SystemExit(f"Environment variable {args.api_key_env} is not set")
        print(json.dumps(judge(args.cases, args.output, base_url=args.base_url, model=args.model,
                               judge_name=args.judge_name, api_key=api_key, max_tokens=args.max_tokens)))
    else:
        rows = score(args.key, args.verdicts, grade=args.grade)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["judge"])
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
