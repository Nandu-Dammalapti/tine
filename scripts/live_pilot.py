#!/usr/bin/env python3
"""A live pilot with real models, as one customer would run it (results/2026-11-17-live-pilot).

``sample``     Disjoint question sets from the local MMLU-Pro and Arena-Hard copies.
``ask``        Ask models a set directly (the customer's graded history). Paid; capped.
               Writes canonical outcomes and an answers CSV for scripts/agreement_pairs.py.
               ``--batch`` sends the same requests through Anthropic's Message Batches API
               instead: one batch per model, at half price, answered within 24 hours
               (most within an hour) and outside the Messages API rate limits. Submitted
               batches are kept in ``<output>.batches.json``, so a rerun collects them
               and never submits the same question twice.
``summarize``  Accuracy, reader match rate, cost and latency per model from ``ask`` output.
               Batch records have no latency.

Answers are read with the proxy's own ``extract_answer``, so offline grading and the
proxy read replies the same way. Nothing here executes model output. Prompts and
replies stay under artifacts/; only summary tables are written for results.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.judge import ANTHROPIC_PRICES  # noqa: E402
from tine.shadow import extract_answer  # noqa: E402

ANSWER_PATTERN = r"Answer:\s*\**\s*\$?\(?([A-J])\b"
# The content of the last \boxed{...}, with up to two levels of braces inside it.
BOXED_PATTERN = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
TASK_TYPE = "mmlupro-live"
TENANT = "live-pilot"
SETS = {"calibrate": 1000, "shadow": 300, "live": 200}
# Each benchmark: where its records are, how the proxy reads an answer, how an answer
# is graded against the key ("exact" letters; "math": equal after normalising the
# LaTeX, or as numbers; "finance": as the benchmark grades FinQA, a number counts when
# it rounds to the key at the key's precision, and words ignore case), and the set sizes. Results files give every model's records
# with the same prompts and keys, so any one model's file will do.
# Message Batches, from Anthropic's batch-processing and rate-limits pages as read on
# 2026-11-17: all usage is billed at half the standard price; a batch holds at most
# 100,000 requests or 256 MB; a custom_id is 1 to 64 letters, digits, hyphens and
# underscores; results come in any order and can be fetched for 29 days.
BATCH_DISCOUNT = 0.5
BATCH_MAX_REQUESTS = 100_000
BATCH_MAX_BYTES = 200 * 1024 * 1024  # below the 256 MB limit, leaving room for the envelope
CUSTOM_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")
DATASETS = {
    "mmlupro": {"folder": "mmlupro/test_3000/claude-sonnet-4", "pattern": ANSWER_PATTERN, "grade": "exact",
                "sets": SETS},
    "medqa": {"folder": "medqa/test", "pattern": ANSWER_PATTERN, "grade": "exact",
              "sets": {"calibrate": 800, "shadow": 200, "live": 150}},
    "arcc": {"folder": "arcc/test", "pattern": r"Answer:\s*\**\s*\$?\(?([A-J1-9])\b", "grade": "exact",
             "sets": {"calibrate": 800, "shadow": 200, "live": 150}},
    "finqa": {"folder": "finqa/test", "pattern": BOXED_PATTERN, "grade": "finance",
              "sets": {"calibrate": 800, "shadow": 200, "live": 140}},
    "math500": {"folder": "math500/test", "pattern": BOXED_PATTERN, "grade": "math",
                "sets": {"calibrate": 330, "shadow": 100, "live": 70}},
}


def _dataset(row: dict) -> str:
    return row.get("dataset", "mmlupro")


def normalise(answer: str) -> str:
    """A LaTeX or numeric answer in one canonical spelling, for grading only."""
    text = answer.strip().strip("$").strip()
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    for old, new in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
                     ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""),
                     ("%", ""), ("\\$", ""), (",\\!", "")):
        text = text.replace(old, new)
    text = "".join(text.split()).rstrip(".")
    return text


def _number(text: str) -> float | None:
    found = re.fullmatch(r"-?\d+(?:\.\d+)?|-?\.\d+", text.replace(",", ""))
    return float(found.group(0)) if found else None


def grade(dataset: str, answer: str | None, truth: str | None) -> bool | None:
    """Whether an answer matches the key. None when the question has no key."""
    if truth is None:
        return None
    if answer is None:
        return False
    if DATASETS[dataset]["grade"] == "exact":
        return answer == truth
    mine, key = normalise(answer), normalise(truth)
    if mine == key:
        return True
    a, b = _number(mine), _number(key)
    if DATASETS[dataset]["grade"] == "finance":
        if a is None or b is None:
            return mine.lower() == key.lower()
        decimals = len(key.replace(",", "").partition(".")[2])
        return abs(a - b) <= 0.5 * 10 ** -decimals + 1e-9
    return a is not None and b is not None and abs(a - b) <= max(1e-6, 1e-4 * abs(b))


def _records(folder: Path) -> list[dict]:
    files = sorted(folder.glob("*.json"))
    if not files:
        raise SystemExit(f"no result JSON under {folder}")
    return json.loads(files[0].read_text())["records"]


def sample_dataset(bench: Path, output: Path, dataset: str, *, seed: int = 0) -> dict:
    """Disjoint calibrate / shadow / live sets from one more benchmark."""
    config = DATASETS[dataset]
    folder = bench / config["folder"]
    if not list(folder.glob("*.json")):
        folder = sorted(path for path in folder.iterdir() if path.is_dir())[0]
    questions = [{"task_id": f"{dataset}:{r['index']}", "prompt": r["prompt"], "ground_truth": str(r["ground_truth"]),
                  "dataset": dataset}
                 for r in _records(folder) if str(r.get("ground_truth") or "").strip()]
    random.Random(seed).shuffle(questions)
    output.mkdir(parents=True, exist_ok=True)
    counts, start = {}, 0
    for name, size in config["sets"].items():
        chosen = questions[start:start + size]
        start += size
        (output / f"{dataset}-{name}.jsonl").write_text("".join(json.dumps(q) + "\n" for q in chosen))
        counts[name] = len(chosen)
    return counts


def sample(bench: Path, output: Path, *, freetext: int = 60, seed: int = 0) -> dict:
    """Disjoint question sets. The probe is the first 40 of the calibrate set."""
    questions = [{"task_id": f"mmlupro:test_3000:{r['index']}", "prompt": r["prompt"],
                  "ground_truth": r["ground_truth"]}
                 for r in _records(bench / "mmlupro" / "test_3000" / "claude-sonnet-4")]
    rng = random.Random(seed)
    rng.shuffle(questions)
    output.mkdir(parents=True, exist_ok=True)
    counts, start = {}, 0
    for name, size in SETS.items():
        chosen = questions[start:start + size]
        start += size
        (output / f"{name}.jsonl").write_text("".join(json.dumps(q) + "\n" for q in chosen))
        counts[name] = len(chosen)
    (output / "probe.jsonl").write_text("".join(
        line + "\n" for line in (output / "calibrate.jsonl").read_text().splitlines()[:40]))
    counts["probe"] = 40
    texts = [{"task_id": f"arenahard-cw:{r['index']}", "prompt": r["prompt"], "ground_truth": None}
             for r in _records(bench / "arenahard_creative_writing" / "claude-sonnet-4")]
    rng.shuffle(texts)
    (output / "freetext.jsonl").write_text("".join(json.dumps(t) + "\n" for t in texts[:freetext]))
    counts["freetext"] = min(freetext, len(texts))
    return counts


def _asked(output: Path) -> tuple[set, float]:
    """The (task, model) pairs an output file already answers, and what they cost."""
    done, spent = set(), 0.0
    if output.exists():
        for line in output.read_text().splitlines():
            if line:
                record = json.loads(line)
                done.add((record["task_id"], record["model"]))
                spent += record["cost_usd"]
    return done, spent


def _client(ca_bundle: str | None):
    import anthropic

    http = None
    if ca_bundle:
        import ssl

        http = anthropic.DefaultHttpxClient(verify=ssl.create_default_context(cafile=ca_bundle))
    return anthropic.Anthropic(max_retries=5, **({"http_client": http} if http else {}))


def _params(row: dict, model: str, max_tokens: int, effort: str) -> dict:
    """The request for one question, the same whether it is sent alone or in a batch."""
    return {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": row["prompt"]}],
            "output_config": {"effort": effort}}


def _worst_case(row: dict, model: str, max_tokens: int) -> float:
    """The most one call can cost at standard prices: a long count of the prompt, a full reply."""
    price_in, price_out = ANTHROPIC_PRICES[model]
    return len(row["prompt"]) / 3 * price_in / 1e6 + max_tokens * price_out / 1e6


def _record(row: dict, model: str, payload: dict, *, latency_ms: int | None, discount: float = 1.0) -> dict:
    """One canonical outcome row, with the answer and reply, from a message payload."""
    price_in, price_out = ANTHROPIC_PRICES[model]
    usage = payload["usage"]
    cost = (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out) / 1e6 * discount
    dataset = _dataset(row)
    answer = extract_answer("anthropic", payload, DATASETS[dataset]["pattern"])
    return {"tenant": TENANT, "task_id": row["task_id"], "prompt": row["prompt"],
            "task_type": f"{dataset}-live", "model": model,
            "success": grade(dataset, answer, row.get("ground_truth")),
            "cost_usd": round(cost, 8), "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"], "cached_tokens": None, "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "answer": answer, "stop_reason": payload.get("stop_reason"),
            "reply": "".join(block.get("text", "") for block in payload["content"]
                             if block.get("type") == "text")}


def _batches_file(output: Path) -> Path:
    return output.with_name(output.name + ".batches.json")


def ask(questions: Path, output: Path, *, models: list[str], max_cost_usd: float, effort: str = "medium",
        max_tokens: int = 4096, workers: int = 8, ca_bundle: str | None = None, client=None) -> dict:
    """Ask every model every question once. Resumes; stops before a call whose worst case
    would pass the cap. ``output`` gets canonical outcome rows plus answer and reply."""
    rows = [json.loads(line) for line in questions.read_text().splitlines() if line]
    done, spent = _asked(output)
    sidecar = _batches_file(output)
    if sidecar.exists() and not all(batch["collected"] for batch in json.loads(sidecar.read_text())["batches"]):
        raise SystemExit(f"{sidecar} has batches still to collect; rerun with --batch first, "
                         "or their questions would be paid for twice")
    if client is None:
        client = _client(ca_bundle)
    state = {"spent": spent, "reserved": 0.0, "asked": 0, "errors": 0, "stopped_at_cap": False}
    lock = threading.Lock()
    output.parent.mkdir(parents=True, exist_ok=True)

    def one(row: dict, model: str) -> None:
        worst = _worst_case(row, model, max_tokens)
        with lock:
            if state["spent"] + state["reserved"] + worst > max_cost_usd:
                state["stopped_at_cap"] = True
                return
            state["reserved"] += worst
        started = time.time()
        try:
            response = client.messages.create(**_params(row, model, max_tokens, effort))
        except Exception as error:  # recorded, never retried silently past the SDK's own retries
            with lock:
                state["reserved"] -= worst
                state["errors"] += 1
            print(f"{model} {row['task_id']}: {type(error).__name__}", file=sys.stderr)
            return
        with lock:
            state["reserved"] -= worst
        record = _record(row, model, response.model_dump(), latency_ms=round((time.time() - started) * 1000))
        with lock:
            state["spent"] += record["cost_usd"]
            state["asked"] += 1
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    pending = [(row, model) for row in rows for model in models if (row["task_id"], model) not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(one, row, model) for row, model in pending]:
            future.result()
    return {"asked": state["asked"], "already_done": len(done), "errors": state["errors"],
            "stopped_at_cap": state["stopped_at_cap"], "spent_usd": round(state["spent"], 4)}


def custom_ids(task_ids: list[str]) -> dict[str, str]:
    """custom_id -> task_id for the requests of one batch. A task id may hold characters
    a custom_id may not (``medqa:123``), so each request is named by its position."""
    return {f"q{index}": task_id for index, task_id in enumerate(task_ids)}


def _chunks(rows: list[dict], weigh) -> list[list[dict]]:
    """Rows split so that no batch passes the request or size limit."""
    chunks, size = [[]], 0
    for row in rows:
        weight = weigh(row)
        if chunks[-1] and (len(chunks[-1]) >= BATCH_MAX_REQUESTS or size + weight > BATCH_MAX_BYTES):
            chunks.append([])
            size = 0
        chunks[-1].append(row)
        size += weight
    return [chunk for chunk in chunks if chunk]


def _write_batches(sidecar: Path, batches: list[dict]) -> None:
    scratch = sidecar.with_name(sidecar.name + ".tmp")
    scratch.write_text(json.dumps({"version": 1, "batches": batches}, indent=1) + "\n")
    scratch.replace(sidecar)


def ask_batch(questions: Path, output: Path, *, models: list[str], max_cost_usd: float, effort: str = "medium",
              max_tokens: int = 4096, ca_bundle: str | None = None, client=None, poll_seconds: float = 60.0,
              timeout_minutes: float = 120.0, sleep=time.sleep) -> dict:
    """``ask`` through the Message Batches API: one batch per model, at the batch price.

    Only requests whose worst case (the direct path's, times the batch discount) fits
    under the cap are submitted, counting what ``output`` records as spent and the worst
    case of every batch not yet collected. Each batch is written to
    ``<output>.batches.json`` before it is created and again with its id before any
    polling, so a rerun polls the same batches and submits only what is in neither the
    output nor an open batch. A create that failed without a clear refusal stays in that
    file with a null id and blocks its questions: look the batch up in the Console, then
    put its id in the entry or delete the entry. Errored, expired and canceled results
    are counted, not written, so a rerun asks them again. After ``timeout_minutes``, or
    a failed call, the run ends with the open batches listed as pending; a rerun
    collects them (results can be fetched for 29 days)."""
    rows = [json.loads(line) for line in questions.read_text().splitlines() if line]
    by_id = {row["task_id"]: row for row in rows}
    done, spent = _asked(output)
    already = len(done)
    sidecar = _batches_file(output)
    batches = json.loads(sidecar.read_text())["batches"] if sidecar.exists() else []
    waiting = [batch for batch in batches if not batch["collected"]]
    unknown = sorted({task for batch in waiting for task in batch["requests"].values()} - set(by_id))
    if unknown:
        raise SystemExit(f"{sidecar} has open batches for questions not in {questions} (first: {unknown[0]}); "
                         "rerun with the questions file they were submitted from")
    claimed = {(task, batch["model"]) for batch in waiting for task in batch["requests"].values()}
    committed = spent + sum(_worst_case(by_id[task], batch["model"], batch["max_tokens"]) * BATCH_DISCOUNT
                            for batch in waiting for task in batch["requests"].values()
                            if (task, batch["model"]) not in done)
    chosen, left_out = {model: [] for model in models}, 0
    for row in rows:
        for model in models:
            pair = (row["task_id"], model)
            if pair in done or pair in claimed:
                continue
            claimed.add(pair)
            worst = _worst_case(row, model, max_tokens) * BATCH_DISCOUNT
            if committed + worst > max_cost_usd:
                left_out += 1
                continue
            committed += worst
            chosen[model].append(row)
    creator = client
    if client is None:
        client = _client(ca_bundle)
        # A retried create could submit the same batch twice, so a create is tried once.
        creator = client.with_options(max_retries=0)
    state = {"spent": spent, "asked": 0, "submitted": 0, "errored": 0, "expired": 0, "canceled": 0,
             "failed_calls": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = []
    for model in models:
        plan += [(model, chunk) for chunk in _chunks(
            chosen[model], lambda row: len(json.dumps(_params(row, model, max_tokens, effort))) + 100)]
    for model, chunk in plan:
        ids = custom_ids([row["task_id"] for row in chunk])
        assert len(ids) == len(chunk) and all(CUSTOM_ID.fullmatch(custom_id) for custom_id in ids)
        # The SDK's Request and MessageCreateParamsNonStreaming are typed dicts: these are the same.
        requests = [{"custom_id": custom_id, "params": _params(row, model, max_tokens, effort)}
                    for custom_id, row in zip(ids, chunk)]
        entry = {"id": None, "model": model, "effort": effort, "max_tokens": max_tokens,
                 "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "worst_case_usd": round(sum(_worst_case(row, model, max_tokens) for row in chunk) * BATCH_DISCOUNT, 6),
                 "requests": ids, "collected": False}
        batches.append(entry)
        _write_batches(sidecar, batches)  # the intent, before the paid call
        try:
            entry["id"] = creator.messages.batches.create(requests=requests).id
        except Exception as error:
            status = getattr(error, "status_code", None)
            if isinstance(status, int) and 400 <= status < 500:  # refused, so nothing was created
                batches.remove(entry)
                _write_batches(sidecar, batches)
            state["failed_calls"] += 1
            print(f"{model} batch of {len(chunk)}: {type(error).__name__}; nothing more submitted", file=sys.stderr)
            break
        _write_batches(sidecar, batches)
        state["submitted"] += len(chunk)

    def collect(entry: dict) -> dict:
        counts = {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
        for item in client.messages.batches.results(entry["id"]):  # any order: matched by custom_id
            line = item.model_dump()
            task, result, model = entry["requests"][line["custom_id"]], line["result"], entry["model"]
            counts[result["type"]] = counts.get(result["type"], 0) + 1
            if result["type"] != "succeeded":
                detail = result.get("error") or {}
                reason = (detail.get("error") or detail).get("type") or result["type"]
                print(f"{model} {task}: {result['type']} ({reason})", file=sys.stderr)
            elif (task, model) not in done:
                record = _record(by_id[task], model, result["message"], latency_ms=None, discount=BATCH_DISCOUNT)
                record["batch_id"] = entry["id"]
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                done.add((task, model))
                state["spent"] += record["cost_usd"]
                state["asked"] += 1
        return counts

    deadline = time.monotonic() + timeout_minutes * 60
    while True:
        for entry in [batch for batch in batches if batch["id"] and not batch["collected"]]:
            try:
                if client.messages.batches.retrieve(entry["id"]).processing_status != "ended":
                    continue
                entry["counts"] = collect(entry)
            except Exception as error:  # the batch stays open; a rerun asks for it again
                state["failed_calls"] += 1
                print(f"{entry['id']}: {type(error).__name__}; rerun to collect", file=sys.stderr)
                break
            entry["collected"] = True
            _write_batches(sidecar, batches)
            for kind in ("errored", "expired", "canceled"):
                state[kind] += entry["counts"][kind]
        pending = [batch["id"] for batch in batches if batch["id"] and not batch["collected"]]
        if not pending or state["failed_calls"] or time.monotonic() >= deadline:
            break
        print(f"waiting on {len(pending)} batch(es)", file=sys.stderr)
        sleep(max(0.0, min(poll_seconds, deadline - time.monotonic())))
    return {"asked": state["asked"], "already_done": already, "submitted": state["submitted"],
            "errors": state["errored"], "expired": state["expired"], "canceled": state["canceled"],
            "failed_calls": state["failed_calls"], "stopped_at_cap": left_out > 0, "left_out_at_cap": left_out,
            "pending_batches": pending, "unconfirmed_batches": sum(batch["id"] is None for batch in batches),
            "spent_usd": round(state["spent"], 4), "batches_file": str(sidecar)}


def export(asked: Path, outcomes: Path, answers: Path) -> dict:
    """Canonical outcomes (without the extra columns) and the answers CSV agreement_pairs reads."""
    records = [json.loads(line) for line in asked.read_text().splitlines() if line]
    keep = ["tenant", "task_id", "prompt", "task_type", "model", "success", "cost_usd", "input_tokens",
            "output_tokens", "cached_tokens", "latency_ms", "timestamp"]
    outcomes.write_text("".join(json.dumps({k: r[k] for k in keep}) + "\n" for r in records))
    with answers.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_id", "model", "answer"])
        for r in records:
            writer.writerow([r["task_id"], r["model"], r["answer"] or ""])
    return {"rows": len(records)}


class RecordingTransport(httpx.AsyncBaseTransport):
    """Passes every call the proxy makes to the provider through unchanged, and keeps
    the model, prompt, reply and token counts of each, so every model's answer to
    every request can be graded afterwards. Also stops new calls past a spending cap."""

    def __init__(self, inner, max_cost_usd: float, pattern: str = ANSWER_PATTERN) -> None:
        self.inner, self.max_cost_usd, self.pattern = inner, max_cost_usd, pattern
        self.calls: list[dict] = []
        self.spent = 0.0
        self.refused = 0

    async def handle_async_request(self, request):
        body = json.loads(request.content)
        model = body.get("model")
        if self.spent >= self.max_cost_usd:
            self.refused += 1
            return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error",
                                                                            "message": "pilot spending cap reached"}})
        started = time.time()
        try:
            response = await self.inner.handle_async_request(request)
        except httpx.HTTPError as error:
            self.calls.append({"model": model, "prompt": body["messages"][-1]["content"], "status": 599,
                               "error": type(error).__name__, "seconds": round(time.time() - started, 2)})
            raise
        raw = await response.aread()
        record = {"model": model, "prompt": body["messages"][-1]["content"], "status": response.status_code,
                  "seconds": round(time.time() - started, 2)}
        if response.status_code < 400:
            payload = json.loads(raw)
            price_in, price_out = ANTHROPIC_PRICES[model]
            usage = payload.get("usage") or {}
            cost = (usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out) / 1e6
            self.spent += cost
            record.update(answer=extract_answer("anthropic", payload, self.pattern), cost_usd=round(cost, 8),
                          stop_reason=payload.get("stop_reason"),
                          reply=extract_answer("anthropic", payload, None))
        self.calls.append(record)
        # aread() decoded the body, so the encoding and length headers no longer apply.
        headers = [(k, v) for k, v in response.headers.items()
                   if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")]
        return httpx.Response(response.status_code, headers=headers, content=raw, request=request)

    async def aclose(self) -> None:
        await self.inner.aclose()


async def through_proxy(policy: dict, questions: list[dict], *, mode: str, max_cost_usd: float, ledger_path: Path,
                        concurrency: int = 6, effort: str = "medium", max_tokens: int = 4096,
                        ca_bundle: str | None = None, inner=None, audit_rate: float = 0.0) -> dict:
    """Send every question to the real proxy as the customer's app would, in shadow mode
    (sample rate 1) or routing through the policy, with the real provider behind it."""
    import asyncio
    import os
    import ssl

    import httpx

    from tine.ledger import Ledger
    from tine.proxy import ProxyConfig, create_app

    frontier = policy["frontier"]
    prices = {f"anthropic:{model}": {"input_per_million": rates[0], "output_per_million": rates[1]}
              for model, rates in ANTHROPIC_PRICES.items()}
    tenant_policy = {TENANT: {"anthropic": policy}}
    config = ProxyConfig(
        tenant_keys={"pilot-token": TENANT}, anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        anthropic_frontier_model=frontier, prices=prices, verified_cache=False,
        **({"route_policies": tenant_policy, "audit_rate": audit_rate} if mode == "route" else
           {"shadow_policies": tenant_policy, "shadow_sample_rate": 1.0, "shadow_max_cost_usd": max_cost_usd}))
    verify = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else True
    # A bare transport ignores HTTPS_PROXY, which httpx clients and the SDK honour.
    network_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    recorder = RecordingTransport(inner or httpx.AsyncHTTPTransport(verify=verify, retries=2, proxy=network_proxy),
                                  max_cost_usd, policy.get("answer_pattern") or ANSWER_PATTERN)
    ledger = Ledger(ledger_path)
    sent: dict[str, dict] = {}
    retries = [0]
    gate = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(transport=recorder, timeout=600) as upstream:
        app = create_app(config, ledger=ledger, client=upstream, policy_client=upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine", timeout=900) as client:
            headers = {"x-api-key": "pilot-token", "anthropic-version": "2023-06-01"}

            async def one(question: dict) -> None:
                async with gate:
                    if recorder.spent >= max_cost_usd:
                        return
                    started = time.time()
                    # As a customer's app would: a rate-limited or overloaded reply is retried
                    # after a pause. Each try is a new request to the proxy with its own ledger row.
                    for attempt in range(6):
                        response = await client.post("/v1/messages", headers=headers, json={
                            "model": frontier, "max_tokens": max_tokens, "output_config": {"effort": effort},
                            "messages": [{"role": "user", "content": question["prompt"]}]})
                        if response.status_code not in (429, 529) or recorder.spent >= max_cost_usd:
                            break
                        retries[0] += 1
                        await asyncio.sleep(min(60.0, float(response.headers.get("retry-after") or 2 ** (attempt + 2))))
                    sent[response.headers.get("x-tine-request-id", question["task_id"])] = {
                        "task_id": question["task_id"], "status": response.status_code,
                        "served_model": response.headers.get("x-tine-model"),
                        "route": response.headers.get("x-tine-route"),
                        "answer": extract_answer("anthropic", response.json(), recorder.pattern)
                        if response.status_code < 400 else None,
                        "client_seconds_including_shadow": round(time.time() - started, 2)}

            # One request first: shadow mode admits one run at a time until a run has
            # finished and its cost is known (ShadowStore.admit).
            await one(questions[0])
            await asyncio.gather(*(one(q) for q in questions[1:]))
            shadow = (await client.get("/v1/shadow", headers=headers)).json()
            savings = (await client.get("/v1/savings", headers=headers)).json()
    for request_id, row in sent.items():
        entry = ledger.get(TENANT, request_id) or {}
        row.update(request_id=request_id, ledger_model=entry.get("model"), cost_usd=entry.get("cost_usd"),
                   routing_cost_usd=entry.get("routing_cost_usd"), latency_ms=entry.get("latency_ms"),
                   holdout=entry.get("holdout"))
        run = ledger._db.execute("SELECT accepted, stage, served_by, matches_frontier, shadow_cost_usd "
                                 "FROM shadow_runs WHERE request_id=?", (request_id,)).fetchone() \
            if mode == "shadow" else None
        row.update(dict(run) if run else {})
    return {"mode": mode, "requests": list(sent.values()), "calls": recorder.calls, "shadow_report": shadow,
            "savings_report": savings, "spent_usd": round(recorder.spent, 4), "refused_at_cap": recorder.refused,
            "client_retries": retries[0]}


def score_proxy(result: dict, questions: list[dict], frontier: str) -> dict:
    """Grade what the proxy did against the known answers."""
    truth = {q["task_id"]: q.get("ground_truth") for q in questions}
    dataset_of = {q["task_id"]: _dataset(q) for q in questions}
    prompt_of = {q["task_id"]: q["prompt"] for q in questions}
    answers = {(c["prompt"], c["model"]): c.get("answer") for c in result["calls"] if c["status"] < 400}
    rows = []
    for r in result["requests"]:
        gold = truth.get(r["task_id"])
        prompt = prompt_of[r["task_id"]]
        front = answers.get((prompt, frontier))
        if result["mode"] == "shadow":
            policy_answer = answers.get((prompt, r.get("served_by"))) if r.get("accepted") else front
        else:
            policy_answer = r["answer"]
        rows.append({"task_id": r["task_id"], "freetext": gold is None, "accepted": bool(r.get("accepted"))
                     if result["mode"] == "shadow" else r.get("ledger_model") != frontier,
                     # A frontier call with no readable answer is wrong; no frontier call is unknown.
                     "holdout": bool(r.get("holdout")),
                     "frontier_right": None if gold is None or (prompt, frontier) not in answers
                     else grade(dataset_of[r["task_id"]], front, gold),
                     "policy_right": grade(dataset_of[r["task_id"]], policy_answer, gold),
                     "latency_ms": r.get("latency_ms"), "cost_usd": r.get("cost_usd"),
                     "routing_cost_usd": r.get("routing_cost_usd"), "shadow_cost_usd": r.get("shadow_cost_usd")})
    import numpy as np

    graded = [row for row in rows if not row["freetext"]]
    free = [row for row in rows if row["freetext"]]
    known = [row for row in graded if row["frontier_right"] is not None]
    summary = {"mode": result["mode"], "requests": len(rows), "graded": len(graded), "freetext": len(free),
               "accepted_share_graded": _share(r["accepted"] for r in graded),
               "freetext_accepted": sum(r["accepted"] for r in free),
               "policy_accuracy": _share(r["policy_right"] for r in graded)}
    if known:
        summary.update(frontier_accuracy=_share(r["frontier_right"] for r in known),
                       loss_rate=_share(r["frontier_right"] and not r["policy_right"] for r in known),
                       gain_rate=_share(r["policy_right"] and not r["frontier_right"] for r in known),
                       compared=len(known))
    held = [r for r in graded if r["holdout"]]
    if held:
        routed = [r for r in graded if not r["holdout"]]

        def mean_cost(group):
            known = [(r["cost_usd"] or 0.0) + (r["routing_cost_usd"] or 0.0) for r in group
                     if r["cost_usd"] is not None and r["routing_cost_usd"] is not None]
            return round(sum(known) / len(known), 6) if known else None
        summary.update(routed_requests=len(routed), holdout_requests=len(held),
                       routed_accepted_share=_share(r["accepted"] for r in routed),
                       routed_accuracy=_share(r["policy_right"] for r in routed),
                       holdout_accuracy=_share(r["policy_right"] for r in held),
                       routed_mean_cost_usd=mean_cost(routed), holdout_mean_cost_usd=mean_cost(held))
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    if latencies:
        summary.update(caller_latency_p50_s=round(float(np.percentile(latencies, 50)) / 1000, 2),
                       caller_latency_p95_s=round(float(np.percentile(latencies, 95)) / 1000, 2))
    calls = result["calls"]
    summary["provider_calls"] = len(calls)
    summary["provider_errors"] = sum(c["status"] >= 400 for c in calls)
    summary["reader_match_rate_by_model"] = {
        model: _share(c.get("answer") is not None for c in calls if c["model"] == model and c["status"] < 400
                      and c["prompt"] in {q["prompt"] for q in questions if q.get("ground_truth")})
        for model in sorted({c["model"] for c in calls})}
    by_model = {}
    for c in calls:
        by_model.setdefault(c["model"], 0.0)
        by_model[c["model"]] += c.get("cost_usd") or 0.0
    summary["spend_by_model_usd"] = {m: round(v, 4) for m, v in by_model.items()}
    summary["spent_usd"] = result["spent_usd"]
    return {"summary": summary, "rows": rows}


def _share(values) -> float | None:
    values = list(values)
    return round(sum(bool(v) for v in values) / len(values), 4) if values else None


def summarize(asked: Path) -> list[dict]:
    import pandas as pd

    frame = pd.DataFrame([json.loads(line) for line in asked.read_text().splitlines() if line])
    table = []
    for model, part in frame.groupby("model"):
        graded = part[part.success.notna()]
        latency = pd.to_numeric(part.latency_ms, errors="coerce").dropna()  # batch records have none
        table.append({"model": model, "replies": len(part),
                      "accuracy": round(graded.success.astype(bool).mean(), 4) if len(graded) else None,
                      "reader_match_rate": round(part.answer.notna().mean(), 4),
                      "cut_off_at_max_tokens": int((part.stop_reason == "max_tokens").sum()),
                      "mean_cost_usd": round(part.cost_usd.mean(), 6), "total_cost_usd": round(part.cost_usd.sum(), 4),
                      "mean_output_tokens": round(part.output_tokens.mean(), 1),
                      "latency_p50_s": round(latency.median() / 1000, 2) if len(latency) else None,
                      "latency_p95_s": round(latency.quantile(0.95) / 1000, 2) if len(latency) else None})
    return table


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    pick = commands.add_parser("sample")
    pick.add_argument("--bench-dir", type=Path, required=True, help="extracted bench-release folder")
    pick.add_argument("--output", type=Path, required=True)
    pick.add_argument("--dataset", choices=sorted(DATASETS),
                      help="sample one more benchmark instead of the original MMLU-Pro and free-text sets")
    run = commands.add_parser("ask")
    run.add_argument("--questions", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--models", nargs="+", required=True)
    run.add_argument("--max-cost-usd", type=float, required=True)
    run.add_argument("--effort", default="medium")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--ca-bundle")
    run.add_argument("--batch", action="store_true",
                     help="use the Message Batches API: half price, answers within 24 hours; rerun to collect")
    run.add_argument("--batch-poll-seconds", type=float, default=60.0)
    run.add_argument("--batch-timeout-minutes", type=float, default=120.0,
                     help="stop waiting after this long; the batches stay recorded and a rerun collects them")
    out = commands.add_parser("export")
    out.add_argument("--asked", type=Path, required=True)
    out.add_argument("--outcomes", type=Path, required=True)
    out.add_argument("--answers", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--asked", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    serve = commands.add_parser("proxy", help="send question sets through the real proxy (paid; capped)")
    serve.add_argument("--policy", type=Path, required=True)
    serve.add_argument("--questions", type=Path, nargs="+", required=True)
    serve.add_argument("--mode", choices=("shadow", "route"), required=True)
    serve.add_argument("--max-cost-usd", type=float, required=True)
    serve.add_argument("--ledger", type=Path, required=True)
    serve.add_argument("--output", type=Path, required=True, help="full result JSON (local only)")
    serve.add_argument("--summary", type=Path, help="summary JSON (safe to publish)")
    serve.add_argument("--concurrency", type=int, default=6)
    serve.add_argument("--audit-rate", type=float, default=0.0,
                       help="route mode: share of requests held out and served by the frontier (TINE_AUDIT_RATE)")
    serve.add_argument("--ca-bundle")
    args = parser.parse_args(argv)
    if args.command == "proxy":
        import asyncio
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        if args.output.exists():
            raise SystemExit(f"{args.output} exists; a proxy run is not resumable, use a new name")
        questions = [json.loads(line) for path in args.questions for line in path.read_text().splitlines() if line]
        random.Random(0).shuffle(questions)  # free text arrives mixed in with the rest, as it would
        policy = json.loads(args.policy.read_text())
        result = asyncio.run(through_proxy(policy, questions, mode=args.mode, max_cost_usd=args.max_cost_usd,
                                           ledger_path=args.ledger, concurrency=args.concurrency,
                                           ca_bundle=args.ca_bundle, audit_rate=args.audit_rate))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, default=str))
        scored = score_proxy(result, questions, policy["frontier"])
        published = {**scored["summary"], "shadow_report": result["shadow_report"],
                     "savings_report": result["savings_report"], "refused_at_cap": result["refused_at_cap"],
                     "client_retries": result.get("client_retries", 0)}
        if args.summary:
            args.summary.write_text(json.dumps(published, indent=2, default=str) + "\n")
        print(json.dumps(published, indent=2, default=str))
        return
    if args.command == "sample":
        print(json.dumps(sample_dataset(args.bench_dir, args.output, args.dataset) if args.dataset
                         else sample(args.bench_dir, args.output)))
    elif args.command == "ask":
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        if args.batch:
            print(json.dumps(ask_batch(args.questions, args.output, models=args.models,
                                       max_cost_usd=args.max_cost_usd, effort=args.effort, ca_bundle=args.ca_bundle,
                                       poll_seconds=args.batch_poll_seconds,
                                       timeout_minutes=args.batch_timeout_minutes)))
            return
        print(json.dumps(ask(args.questions, args.output, models=args.models, max_cost_usd=args.max_cost_usd,
                             effort=args.effort, workers=args.workers, ca_bundle=args.ca_bundle)))
    elif args.command == "export":
        print(json.dumps(export(args.asked, args.outcomes, args.answers)))
    else:
        table = summarize(args.asked)
        if args.output:
            import pandas as pd

            pd.DataFrame(table).to_csv(args.output, index=False)
        print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
