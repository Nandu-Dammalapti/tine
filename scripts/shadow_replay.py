#!/usr/bin/env python3
"""Replay recorded answers through the proxy's shadow mode or its live routing.

A wiring check and a demonstration, with no model called: a stand-in provider answers
every request from recorded benchmark responses (the answer each model gave, with its
recorded token counts), the real proxy app serves the requests with a frozen policy in
shadow mode, and the shadow report is read back from its own endpoint. Because the
grades are known, the script also says how often the policy's accepted answers were
right, which a live shadow report cannot know.

By default it replays the tasks of the policy's own held-out test part, so the
projected savings can be set beside the estimate stored in the policy file. The two
differ slightly by design: the stored estimate charges a model once per stage it
appears in, and a live run calls it once per request.

``--mode route`` serves the same requests from the policy instead (live routing) and
reads each request's served model, answer cost and routing cost back from the ledger.

Replay requires one tenant's known grades, recorded answers and token counts for
every requested task and policy model. Missing evidence is an error, not a failed
answer or an assumed frontier result.
When a policy was frozen for a different deployment tenant, its recorded
``provenance.outcomes_tenant`` also identifies valid source outcomes; the replay
still runs under the policy's deployment tenant.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.data import _validate_csv_structure, read_outcomes, validate_outcomes  # noqa: E402
from tine.ledger import Ledger  # noqa: E402
from tine.pipeline import complete_tasks  # noqa: E402
from tine.policy import validate_policy  # noqa: E402
from tine.proxy import ProxyConfig, create_app  # noqa: E402
from tine.shadow import WHOLE_REPLY_PATTERN  # noqa: E402


def test_part(group_outcomes, frontier: str, seed: int, fractions) -> set[str]:
    """Task ids of the test part, split exactly as ``tine.policy.freeze_policy`` does."""
    _, tasks, _ = complete_tasks(group_outcomes, frontier)
    part = np.random.default_rng(seed).choice(3, size=len(tasks), p=list(fractions))
    return set(tasks.task_id[part == 2])


def read_answers(path: Path) -> dict:
    """Read one recorded answer per task/model without overwriting duplicate rows."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        _validate_csv_structure(stream)
        reader = csv.DictReader(stream)
        if not {"task_id", "model", "answer"}.issubset(reader.fieldnames or []):
            raise ValueError("Replay answers require task_id, model, and answer columns")
        answers = {}
        for row in reader:
            key = (row["task_id"], row["model"])
            if not all(value.strip() for value in key):
                raise ValueError("Replay answer identifiers must be nonempty")
            if key in answers:
                raise ValueError(f"Duplicate recorded answer for {key}; choose one run explicitly")
            answers[key] = row["answer"]
    return answers


async def replay(outcomes, answers: dict, policy: dict, prices: dict, task_ids: list[str],
                 mode: str = "shadow") -> dict:
    outcomes = validate_outcomes(outcomes, require_known=True)
    validate_policy(policy)
    if mode not in {"shadow", "route"}:
        raise ValueError("Replay mode must be shadow or route")
    if outcomes.tenant.nunique() != 1:
        raise ValueError("Replay outcomes must belong to one tenant")
    recorded_tenant = outcomes.tenant.iloc[0]
    tenant, frontier = policy.get("tenant") or recorded_tenant, policy["frontier"]
    provenance = policy.get("provenance")
    source_tenant = provenance.get("outcomes_tenant") if isinstance(provenance, dict) else None
    if recorded_tenant != tenant and recorded_tenant != source_tenant:
        raise ValueError("Replay outcomes must match the policy tenant or its recorded source tenant")
    task_ids = list(task_ids)
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Replay task IDs must be a nonempty unique selection")
    models = {frontier, *(model for stage in policy["stages"] for model in stage["models"])}
    outcomes = outcomes.loc[outcomes.task_id.isin(task_ids) & outcomes.model.isin(models)]
    rows = {(row.task_id, row.model): row for row in outcomes.itertuples()}
    for task_id in task_ids:
        for model in sorted(models):
            key = (task_id, model)
            if key not in rows:
                raise ValueError(f"Missing recorded outcome for {key}")
            if key not in answers or not isinstance(answers[key], str):
                raise ValueError(f"Missing recorded answer text for {key}")
            if model not in prices:
                raise ValueError(f"Missing replay prices for model {model!r}")
    if outcomes[["input_tokens", "output_tokens"]].isna().any().any():
        raise ValueError("Replay requires known input_tokens and output_tokens")
    prompts = {row.prompt: row.task_id for row in outcomes.drop_duplicates("task_id").itertuples()}
    if len(prompts) != outcomes.task_id.nunique():
        raise ValueError("Prompts must identify tasks uniquely for a replay")

    def provider(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        task_id = prompts[body["messages"][0]["content"]]
        row = rows[(task_id, body["model"])]
        answer = answers[(task_id, body["model"])]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": answer}}],
            "usage": {"prompt_tokens": int(row.input_tokens), "completion_tokens": int(row.output_tokens)}})

    config = ProxyConfig(
        tenant_keys={"replay-token": tenant}, openai_api_key="replay", openai_frontier_model=frontier,
        prices={f"openai:{model}": rates for model, rates in prices.items()}, verified_cache=False,
        # Each stand-in reply is a recorded answer and nothing else, so the whole reply is
        # the answer, as it was when agreement was measured.
        answer_pattern=WHOLE_REPLY_PATTERN,
        **({"route_policies": {tenant: {"openai": policy}}} if mode == "route" else
           {"shadow_policies": {tenant: {"openai": policy}}, "shadow_sample_rate": 1.0, "shadow_max_cost_usd": 1e9}))
    ledger = Ledger()
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
        app = create_app(config, ledger=ledger, client=upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
            headers = {"authorization": "Bearer replay-token"}
            request_tasks = {}
            for task_id in task_ids:
                prompt = rows[(task_id, frontier)].prompt
                response = await client.post("/v1/chat/completions", headers=headers,
                                             json={"model": frontier, "messages": [{"role": "user", "content": prompt}]})
                response.raise_for_status()
                request_tasks[response.headers["x-tine-request-id"]] = task_id
            report = (await client.get("/v1/shadow", headers=headers)).json() if mode == "shadow" else None
    if mode == "route":
        return route_summary(ledger, tenant, request_tasks, rows, frontier, prices, policy)
    # What a live deployment cannot see: whether each path's answer was right.
    served = {row["request_id"]: row for row in ledger._db.execute("SELECT request_id, accepted, served_by FROM shadow_runs")}
    policy_right = frontier_right = lost = 0
    for request_id, task_id in request_tasks.items():
        run = served.get(request_id)
        front = bool(rows[(task_id, frontier)].success)
        mine = bool(rows[(task_id, run["served_by"])].success) if run is not None and run["accepted"] else front
        policy_right += mine
        frontier_right += front
        lost += front and not mine
    count = len(request_tasks)
    return {"requests": count, "shadow_report": report,
            "graded": {"policy_accuracy": policy_right / count, "frontier_accuracy": frontier_right / count,
                       "degradation_rate": lost / count},
            "policy_estimate": policy.get("estimate")}


def route_summary(ledger, tenant, request_tasks, rows, frontier, prices, policy) -> dict:
    """What live routing did, from the ledger, against the frontier's recorded cost."""
    rates = prices[frontier]
    spent = baseline = 0.0
    right = frontier_right = lost = routed = 0
    for request_id, task_id in request_tasks.items():
        entry = ledger.get(tenant, request_id)
        front = rows[(task_id, frontier)]
        spent += entry["cost_usd"] + entry["routing_cost_usd"]
        baseline += (front.input_tokens * rates["input_per_million"]
                     + front.output_tokens * rates["output_per_million"]) / 1e6
        mine = bool(rows[(task_id, entry["model"])].success)
        routed += entry["model"] != frontier
        right += mine
        frontier_right += bool(front.success)
        lost += bool(front.success) and not mine
    count = len(request_tasks)
    return {"requests": count, "mode": "route",
            "ledger": {"served_by_cheaper_models_share": routed / count, "cost_usd": spent,
                       "frontier_cost_usd": baseline, "savings_fraction": 1 - spent / baseline if baseline else None},
            "graded": {"policy_accuracy": right / count, "frontier_accuracy": frontier_right / count,
                       "degradation_rate": lost / count},
            "policy_estimate": policy.get("estimate")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outcomes", type=Path, required=True, help="canonical outcomes, one row per task and model")
    parser.add_argument("--answers", type=Path, required=True, help="CSV (task_id, model, answer)")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True,
                        help="JSON: model -> {input_per_million, output_per_million}")
    parser.add_argument("--group-outcomes", type=Path,
                        help="the outcomes the policy was frozen from; replays only its test part")
    parser.add_argument("--mode", choices=("shadow", "route"), default="shadow")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    outcomes = read_outcomes(args.outcomes)
    answers = read_answers(args.answers)
    task_ids = sorted(outcomes.task_id.unique())
    if args.group_outcomes:
        provenance = policy["provenance"]
        keep = test_part(read_outcomes(args.group_outcomes), policy["frontier"], provenance["seed"],
                         provenance["fractions"])
        task_ids = [task_id for task_id in task_ids if task_id in keep]
    result = asyncio.run(replay(outcomes, answers, policy, json.loads(args.prices.read_text()),
                                task_ids[:args.limit], mode=args.mode))
    text = json.dumps(result, indent=2, default=float)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
