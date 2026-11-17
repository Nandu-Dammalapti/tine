import asyncio
import importlib.util
import json
import os
import signal
import time
from pathlib import Path

import httpx
import pytest

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
    assert result["stopped_by"] is None and result["abandoned_in_flight"] == 0


def signalled(name: str, at: int, seconds: float):
    """A provider that sends this process a signal on its ``at``-th request, as an operator
    would part way through a load, and takes ``seconds`` to answer each request."""
    seen = [0]

    async def handler(request: httpx.Request) -> httpx.Response:
        seen[0] += 1
        if seen[0] == at:
            os.kill(os.getpid(), getattr(signal, name))
        await asyncio.sleep(seconds)
        return httpx.Response(200, json={"usage": {"input_tokens": 20, "output_tokens": 300}})
    return handler


@pytest.mark.parametrize("name", ["SIGINT", "SIGTERM"])
def test_load_stops_on_a_signal_and_still_gives_its_totals(monkeypatch, name):
    monkeypatch.setattr(probe, "new_client",
                        lambda ca: httpx.AsyncClient(transport=httpx.MockTransport(signalled(name, 9, 0.01))))
    before = signal.getsignal(getattr(signal, name))
    started = time.monotonic()
    result = asyncio.run(probe.load(4, seconds=60, max_cost_usd=5.0, ca_bundle=None))
    assert time.monotonic() - started < 10
    # Requests in flight at the signal finish and are counted; none starts after it.
    assert result["stopped_by"] == name and 9 <= result["accepted"] <= 12 and result["abandoned_in_flight"] == 0
    assert result["spent_usd"] == round(result["accepted"] * (20 * 0.10 + 300 * 0.50) / 1e6, 4)
    assert signal.getsignal(getattr(signal, name)) is before


def test_load_drops_requests_that_outlast_the_grace_time(monkeypatch):
    monkeypatch.setattr(probe, "new_client",
                        lambda ca: httpx.AsyncClient(transport=httpx.MockTransport(signalled("SIGTERM", 3, 60))))
    started = time.monotonic()
    result = asyncio.run(probe.load(3, seconds=60, max_cost_usd=5.0, ca_bundle=None, grace_seconds=0.2))
    assert time.monotonic() - started < 10
    assert (result["stopped_by"], result["accepted"], result["abandoned_in_flight"]) == ("SIGTERM", 0, 3)


def test_load_command_prints_its_totals_after_a_signal(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(probe, "new_client",
                        lambda ca: httpx.AsyncClient(transport=httpx.MockTransport(signalled("SIGTERM", 5, 0.01))))
    probe.main(["load", "--in-flight", "2", "--seconds", "60", "--max-cost-usd", "5",
                "--output", str(tmp_path / "load.json")])
    printed = json.loads(capsys.readouterr().out)
    assert printed["stopped_by"] == "SIGTERM" and printed["accepted"] >= 5
    assert json.loads((tmp_path / "load.json").read_text()) == printed
