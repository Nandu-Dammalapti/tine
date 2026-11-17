import asyncio
import importlib.util
from pathlib import Path

import httpx

SPEC = importlib.util.spec_from_file_location(
    "concurrency_probe", Path(__file__).resolve().parents[1] / "scripts" / "concurrency_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def provider(limit: int):
    """Accepts ``limit`` requests at a time and refuses the rest, as the account does."""
    state = {"in_flight": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if state["in_flight"] >= limit:
            return httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit_error",
                                                                          "message": "Number of concurrent requests"}})
        state["in_flight"] += 1
        await asyncio.sleep(0.01)
        state["in_flight"] -= 1
        return httpx.Response(200, json={"usage": {"input_tokens": 20, "output_tokens": 300}})
    return handler


def test_ramp_counts_what_was_accepted_at_once(monkeypatch):
    monkeypatch.setattr(probe, "new_client", lambda ca: httpx.AsyncClient(transport=httpx.MockTransport(provider(5))))
    table = asyncio.run(probe.ramp([4, 9], 1.0, None, pause_seconds=0.05))
    assert [(row["accepted"], row["refused_429"]) for row in table] == [(4, 0), (5, 4)]


def test_load_stops_at_its_spending_cap(monkeypatch):
    monkeypatch.setattr(probe, "new_client", lambda ca: httpx.AsyncClient(transport=httpx.MockTransport(provider(50))))
    result = asyncio.run(probe.load(4, seconds=30, max_cost_usd=0.001, ca_bundle=None))
    assert result["accepted"] >= 4 and result["spent_usd"] < 0.003
