#!/usr/bin/env python3
"""Can the judge model write a task's real fix from the issue alone?

A memory probe for the code judge. For each SWE-bench task the model is shown only
the issue and asked for the change the maintainers merged, as a diff. Its changed
lines are compared with the reference fix. Where it reproduces most of them, it very
likely remembers the fix, and its verdicts on patches for that task may rest on
memory rather than reading. Splitting the judge's measured accuracy by this label
estimates how much of it would carry over to code the model has never seen.

``run``    Ask the model (paid; capped; the key comes from the environment).
``score``  Overlap of each reply with the reference fix, and the judge's accuracy on
           tasks it recalls against tasks it does not, overall and within the
           benchmark's difficulty bands (easy tasks are both easier to recall and
           easier to judge, so the overall split alone would overstate memory).

Nothing here executes code. Replies stay under artifacts/; only summary tables are
written for results.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.judge import ANTHROPIC_PRICES, MAX_ISSUE_CHARS  # noqa: E402

PROMPT_VERSION = "2026-11-16-recall"
INSTRUCTIONS = """Below is a GitHub issue from the {repository} repository. Write the code change the project's maintainers merged to fix it, as a unified diff against the repository as it was when the issue was filed. If you do not know the merged change, write the fix you think is right.

Reply with the diff only, inside one ```diff block."""
RECALL_THRESHOLD = 0.5  # fixed before any reply was read


def changed_lines(diff: str) -> set[str]:
    """Added and removed source lines, whitespace-normalised, without file headers,
    blank lines, or lines that are only punctuation."""
    found = set()
    for line in diff.splitlines():
        if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
            continue
        text = " ".join(line[1:].split())
        if text and re.search(r"[A-Za-z0-9]", text):
            found.add(line[0] + text)
    return found


def overlap(reply: str, reference: str) -> float:
    """Share of the reference fix's changed lines that the reply reproduces."""
    ref = changed_lines(reference)
    return len(changed_lines(reply) & ref) / len(ref) if ref else 0.0


def extract_diff(text: str) -> str:
    blocks = re.findall(r"```(?:diff)?\n(.*?)```", text, flags=re.DOTALL)
    return blocks[-1] if blocks else text


def run(dataset: Path, instance_ids: list[str], output: Path, *, model: str, max_cost_usd: float,
        effort: str = "medium", max_tokens: int = 8000, workers: int = 8, ca_bundle: str | None = None,
        client=None) -> dict:
    price_in, price_out = ANTHROPIC_PRICES[model]
    rows = {json.loads(line)["instance_id"]: json.loads(line) for line in dataset.read_text().splitlines() if line}
    done = set()
    if output.exists():
        done = {json.loads(line)["instance_id"] for line in output.read_text().splitlines() if line}
    spent = sum(json.loads(line)["cost_usd"] for line in output.read_text().splitlines() if line) \
        if output.exists() else 0.0
    if client is None:
        import anthropic

        http = None
        if ca_bundle:
            import ssl

            http = anthropic.DefaultHttpxClient(verify=ssl.create_default_context(cafile=ca_bundle))
        client = anthropic.Anthropic(max_retries=5, **({"http_client": http} if http else {}))
    state = {"spent": spent, "reserved": 0.0, "asked": 0, "stopped_at_cap": False}
    lock = threading.Lock()
    output.parent.mkdir(parents=True, exist_ok=True)

    def one(instance_id: str) -> None:
        row = rows[instance_id]
        system = INSTRUCTIONS.format(repository=row["repo"])
        worst = (len(system) + len(row["problem_statement"][:MAX_ISSUE_CHARS])) / 3 * price_in / 1e6 \
            + max_tokens * price_out / 1e6
        with lock:
            if state["spent"] + state["reserved"] + worst > max_cost_usd:
                state["stopped_at_cap"] = True
                return
            state["reserved"] += worst
        started = time.time()
        try:
            response = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": row["problem_statement"][:MAX_ISSUE_CHARS]}],
                output_config={"effort": effort})
        finally:
            with lock:
                state["reserved"] -= worst
        cost = (response.usage.input_tokens * price_in + response.usage.output_tokens * price_out) / 1e6
        text = "".join(block.text for block in response.content if block.type == "text")
        record = {"instance_id": instance_id, "model": model, "prompt_version": PROMPT_VERSION,
                  "stop_reason": response.stop_reason, "reply": extract_diff(text), "cost_usd": round(cost, 6),
                  "seconds": round(time.time() - started, 2)}
        with lock:
            state["spent"] += cost
            state["asked"] += 1
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    pending = [instance_id for instance_id in instance_ids if instance_id not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(one, instance_id) for instance_id in pending]:
            future.result()
    return {"asked": state["asked"], "skipped": len(done), "stopped_at_cap": state["stopped_at_cap"],
            "spent_usd": round(state["spent"], 4)}


def score(dataset: Path, replies: Path, key: Path, verdicts: Path, output: Path | None = None) -> dict:
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    rows = {json.loads(line)["instance_id"]: json.loads(line) for line in dataset.read_text().splitlines() if line}
    recall = {}
    for line in replies.read_text().splitlines():
        if line:
            record = json.loads(line)
            recall[record["instance_id"]] = overlap(record["reply"], rows[record["instance_id"]]["patch"])
    with key.open(newline="") as stream:
        cases = pd.DataFrame(list(csv.DictReader(stream)))
    cases = cases.merge(pd.read_csv(verdicts)[["case_id", "verdict", "probability"]], on="case_id")
    cases = cases[cases.instance_id.isin(recall)].copy()
    cases["right"] = cases.run_resolved == "True"
    cases["accepted"] = cases.verdict == "accept"
    cases["overlap"] = cases.instance_id.map(recall)
    cases["recalled"] = cases.overlap >= RECALL_THRESHOLD
    cases["difficulty"] = cases.instance_id.map(lambda iid: rows[iid].get("difficulty") or "unknown")

    def summary(part) -> dict:
        right, wrong = part[part.right], part[~part.right]
        accepted = part[part.accepted]
        both = part.right.nunique() == 2
        return {"cases": len(part), "tasks": part.instance_id.nunique(), "right_share": round(part.right.mean(), 4),
                "catch_rate": round((~wrong.accepted).mean(), 4) if len(wrong) else None,
                "false_reject_rate": round((~right.accepted).mean(), 4) if len(right) else None,
                "accuracy_when_accepted": round(accepted.right.mean(), 4) if len(accepted) else None,
                "auc": round(roc_auc_score(part.right, part.probability.fillna(0)), 4) if both else None}

    table = []
    for band in ["all", *sorted(cases.difficulty.unique())]:
        part = cases if band == "all" else cases[cases.difficulty == band]
        for label, chosen in (("recalled", part[part.recalled]), ("not recalled", part[~part.recalled])):
            if len(chosen):
                table.append({"difficulty": band, "group": label, **summary(chosen)})
    tasks = pd.Series(recall)
    result = {"tasks": len(tasks), "recall_threshold": RECALL_THRESHOLD,
              "tasks_recalled": int((tasks >= RECALL_THRESHOLD).sum()),
              "median_overlap": round(float(tasks.median()), 4),
              "overlap_quartiles": [round(float(q), 4) for q in tasks.quantile([0.25, 0.5, 0.75])],
              "tasks_by_difficulty": {f"{band} / {'recalled' if flag else 'not recalled'}": int(count)
                                      for (band, flag), count in cases.drop_duplicates("instance_id")
                                      .groupby(["difficulty", "recalled"]).size().items()},
              "table": table}
    if output:
        pd.DataFrame(table).to_csv(output, index=False)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("run")
    ask.add_argument("--dataset", type=Path, required=True)
    ask.add_argument("--key", type=Path, required=True, help="judge key; its tasks are the ones asked")
    ask.add_argument("--output", type=Path, required=True)
    ask.add_argument("--model", required=True)
    ask.add_argument("--max-cost-usd", type=float, required=True)
    ask.add_argument("--workers", type=int, default=8)
    ask.add_argument("--ca-bundle")
    grade = commands.add_parser("score")
    grade.add_argument("--dataset", type=Path, required=True)
    grade.add_argument("--replies", type=Path, required=True)
    grade.add_argument("--key", type=Path, required=True)
    grade.add_argument("--verdicts", type=Path, required=True)
    grade.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        with args.key.open(newline="") as stream:
            ids = sorted({row["instance_id"] for row in csv.DictReader(stream)})
        print(json.dumps(run(args.dataset, ids, args.output, model=args.model, max_cost_usd=args.max_cost_usd,
                             workers=args.workers, ca_bundle=args.ca_bundle)))
    else:
        result = score(args.dataset, args.replies, args.key, args.verdicts, args.output)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
