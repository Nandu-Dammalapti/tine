#!/usr/bin/env python3
"""Find a provider account's limit on requests in flight, and hold a load near it.

``ramp``  Send N requests at once for rising N and count how many are refused (429),
          how long a refusal takes to come back and whether it names a wait.
``load``  Keep K requests in flight for a while, so another process can be tested
          with only a few slots free (results/2026-11-17-concurrency-and-long-shadow).

Paid, but small: every request is ``claude-haiku-5-5`` writing a few hundred tokens.
Both stop at ``--max-cost-usd``, counted from the API's own token counts. The key
comes from the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import ssl
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.judge import ANTHROPIC_PRICES  # noqa: E402

MODEL = "claude-haiku-5-5"
URL = "https://api.anthropic.com/v1/messages"
BODY = {"model": MODEL, "max_tokens": 300,
        "messages": [{"role": "user", "content": "Count from 1 to 140, one number per line. Nothing else."}]}


class Spend:
    def __init__(self, cap: float) -> None:
        self.cap, self.spent = cap, 0.0

    def add(self, payload: dict) -> None:
        usage = payload.get("usage") or {}
        price_in, price_out = ANTHROPIC_PRICES[MODEL]
        self.spent += (usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out) / 1e6


async def one(client: httpx.AsyncClient, spend: Spend) -> dict:
    started = time.monotonic()
    try:
        response = await client.post(URL, json=BODY)
    except httpx.HTTPError as error:
        return {"status": 599, "seconds": round(time.monotonic() - started, 2), "error": type(error).__name__}
    row = {"status": response.status_code, "seconds": round(time.monotonic() - started, 2),
           "retry_after": response.headers.get("retry-after")}
    if response.status_code < 400:
        spend.add(response.json())
    else:
        row["message"] = (response.json().get("error") or {}).get("message", "")[:120]
    return row


def new_client(ca_bundle: str | None) -> httpx.AsyncClient:
    verify = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else True
    return httpx.AsyncClient(
        verify=verify, timeout=120, limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})


async def ramp(levels: list[int], max_cost_usd: float, ca_bundle: str | None,
               pause_seconds: float = 5.0) -> list[dict]:
    spend, table = Spend(max_cost_usd), []
    async with new_client(ca_bundle) as client:
        for level in levels:
            if spend.spent >= max_cost_usd:
                break
            rows = await asyncio.gather(*(one(client, spend) for _ in range(level)))
            refused = [row for row in rows if row["status"] == 429]
            accepted = [row for row in rows if row["status"] < 400]
            table.append({
                "sent_at_once": level, "accepted": len(accepted), "refused_429": len(refused),
                "other": level - len(accepted) - len(refused),
                "accepted_seconds_median": sorted(r["seconds"] for r in accepted)[len(accepted) // 2] if accepted else None,
                "refusal_seconds_median": sorted(r["seconds"] for r in refused)[len(refused) // 2] if refused else None,
                "retry_after_values": sorted({r["retry_after"] for r in refused if r["retry_after"]}),
                "message": refused[0]["message"] if refused else None,
                "spent_usd": round(spend.spent, 4)})
            print(json.dumps(table[-1]), flush=True)
            await asyncio.sleep(pause_seconds)  # let every slot free before the next level
    return table


async def load(in_flight: int, seconds: float, max_cost_usd: float, ca_bundle: str | None) -> dict:
    """Keep ``in_flight`` requests going. A refused worker waits half a second and asks
    again, so the load presses on the limit the way busy real traffic would."""
    spend = Spend(max_cost_usd)
    counts = {"accepted": 0, "refused_429": 0, "other": 0}
    deadline = time.monotonic() + seconds

    async with new_client(ca_bundle) as client:
        async def worker() -> None:
            while time.monotonic() < deadline and spend.spent < max_cost_usd:
                row = await one(client, spend)
                key = "accepted" if row["status"] < 400 else "refused_429" if row["status"] == 429 else "other"
                counts[key] += 1
                if key != "accepted":
                    await asyncio.sleep(0.5)
        await asyncio.gather(*(worker() for _ in range(in_flight)))
    return {"in_flight": in_flight, "seconds": seconds, **counts, "spent_usd": round(spend.spent, 4)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("ramp")
    up.add_argument("--levels", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64, 96, 128])
    hold = commands.add_parser("load")
    hold.add_argument("--in-flight", type=int, required=True)
    hold.add_argument("--seconds", type=float, required=True)
    for sub in (up, hold):
        sub.add_argument("--max-cost-usd", type=float, required=True)
        sub.add_argument("--ca-bundle")
        sub.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    if args.command == "ramp":
        result = asyncio.run(ramp(args.levels, args.max_cost_usd, args.ca_bundle))
    else:
        result = asyncio.run(load(args.in_flight, args.seconds, args.max_cost_usd, args.ca_bundle))
        print(json.dumps(result))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
