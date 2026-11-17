#!/usr/bin/env python3
"""Measure what the Tine proxy adds to a request: latency, throughput, memory, ledger growth.

``provider``  A stand-in provider on a local port. It speaks the two wire formats the
              proxy forwards, Anthropic ``POST /v1/messages`` and OpenAI ``POST
              /v1/chat/completions``, buffered and streamed, with settable latency,
              error replies and very slow replies. Every reply ends with a line
              ``Answer: X`` so that agreement policies can run. ``--ceiling`` measures
              how fast the stand-in itself is.
``run``       A load generator against any base URL: closed loop (``--in-flight``) or
              open loop (``--rate``, with latency counted from the scheduled send time).
``compare``   One scenario: the same load at the stand-in directly and through the
              proxy, in alternating rounds, and what the proxy added.
``matrix``    The named scenarios in order, one JSON file each and a combined CSV.

What the numbers mean, and what they do not:

- The provider is a stand-in on the same machine. No real model is called, nothing is
  paid and no request leaves the machine. The stand-in's latency is a setting, so a
  figure here says what the proxy adds to a reply, never how long a model takes.
- "Added latency" is the difference between two measured distributions (through the
  proxy, and at the stand-in directly) over loopback. A real deployment adds a network
  hop to the proxy and TLS to the provider, which are not in it.
- The generator, the stand-in and the proxy share one machine's cores. On a laptop, and
  above all on a busy one, they slow each other. Every summary carries the load average
  at its start and end and sets ``busy_machine`` when it was high: such a run shows that
  the tool works and is not to be reported as a measurement.
- Throughput is that of one ``python -m tine serve`` process with a SQLite ledger on
  local disk, as shipped (access log on). It is not a capacity promise for a
  deployment, and says nothing about several processes or another datastore.
- The policy used by the shadow and routing scenarios is synthetic: tenant and task
  type ``load-test``, model names of the stand-in, and guarantee fields that are only
  placeholders the proxy's validator requires. Nothing was certified. Its acceptance
  share is a setting of the stand-in (``--agree-share``), so savings and quality
  figures from these runs mean nothing; only timings and counts do.
- Peak memory and CPU time of the proxy come from the child's resource usage when it
  exits (``wait4``); memory during a run is read from the operating system for that one
  process. Neither is available for a proxy started elsewhere (``--base-url``).

The prompts, replies and ledgers these runs make are synthetic and stay in a temporary
folder that is removed at the end (``--keep`` leaves it). Per-request rows are written
only when asked for (``--rows``). macOS and Linux only.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import resource
import shlex
import shutil
import signal
import socket
import sqlite3
import ssl
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]

TENANT = "load-test"
TASK_TYPE = "load-test"
TOKEN = "load-test-tenant-token"
# Sent to the stand-in as the provider key. Obviously not a credential.
FAKE_KEY = "fake-provider-key-for-load-test-only"
FRONTIER = "standin-frontier"
CHEAP = ("standin-small-a", "standin-small-b")
LETTERS = "ABCDEFGHIJ"
ANSWER_PATTERN = r"Answer:\s*([A-J])\b"
# Made-up prices, so that the proxy records a cost for every stand-in reply.
PRICES = {f"{provider}:{model}": {"input_per_million": rate, "output_per_million": 5 * rate}
          for provider in ("anthropic", "openai")
          for model, rate in ((FRONTIER, 10.0), (CHEAP[0], 1.0), (CHEAP[1], 0.5))}
PORTS = (18000, 18999)
ROUTES = {"anthropic": "/v1/messages", "openai": "/v1/chat/completions"}
CAVEAT = ("A stand-in provider on the same machine, one proxy process, loopback only. Not a capacity promise; "
          "see the module docstring of scripts/load_test.py for what these numbers mean")
BUSY = ("The machine was busy during this run (load average above half its cores), so the generator, the stand-in "
        "and the proxy were short of CPU: this shows the tool works and is not a measurement to report")


def say(*parts) -> None:
    print(*parts, file=sys.stderr, flush=True)


# --- arithmetic ---------------------------------------------------------------------


def percentile(values, percent: float) -> float | None:
    """Nearest rank: the smallest measured value with at least ``percent`` of the values
    at or below it. Never a value between two measurements."""
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(1, math.ceil(round(percent * len(ordered) / 100, 9))) - 1]


def distribution(values) -> dict | None:
    """Count, extremes, mean and nearest-rank percentiles of ``values``, or None for none."""
    ordered = sorted(values)
    if not ordered:
        return None
    count = len(ordered)

    def at(percent: float) -> float:
        return round(ordered[max(1, math.ceil(round(percent * count / 100, 9))) - 1], 3)
    return {"n": count, "min": round(ordered[0], 3), "p50": at(50), "p90": at(90), "p95": at(95), "p99": at(99),
            "max": round(ordered[-1], 3), "mean": round(math.fsum(ordered) / count, 3)}


def schedule(rate: float, count: int) -> list[float]:
    """When an open loop sends each request, in seconds from its start: one every 1/rate,
    whatever happened to the requests before it."""
    return [index / rate for index in range(count)]


def timings(scheduled: float, sent: float, first: float | None, done: float) -> dict:
    """One request's times in milliseconds, from clock readings in seconds.

    Latency and time to first byte run from the *scheduled* send. In a closed loop that
    is the actual send. In an open loop a request that had to wait (for a free
    connection, or for the generator) is charged for the wait, so queueing in front of
    a slow server is not hidden; ``service_ms`` is the part after the actual send."""
    return {"latency_ms": (done - scheduled) * 1000, "service_ms": (done - sent) * 1000,
            "send_delay_ms": (sent - scheduled) * 1000,
            "ttfb_ms": None if first is None else (first - scheduled) * 1000}


def spread(values) -> dict | None:
    """The median of per-round values and how far the rounds were apart."""
    known = [value for value in values if value is not None]
    if not known:
        return None
    return {"median": round(statistics.median(known), 3), "min": round(min(known), 3), "max": round(max(known), 3),
            "rounds": [None if value is None else round(value, 3) for value in values]}


def _unit(digest: bytes, at: int) -> float:
    """A number in [0, 1) from eight bytes of a digest."""
    return int.from_bytes(digest[at:at + 8], "big") / 2**64


def busy(loads) -> bool:
    """Whether the one-minute load average says other work had more than half the cores."""
    return loads[0] > 0.5 * (os.cpu_count() or 1)


# --- the stand-in provider ----------------------------------------------------------


@dataclass
class StandIn:
    """Settings of the stand-in provider. Everything about a reply is decided from the
    request and ``seed`` alone, so the same request gets the same reply, latency and
    status whether it comes directly or through the proxy."""
    latency_ms: float = 0.0  # every reply takes this long, plus a share of the spread
    latency_spread_ms: float = 0.0
    ttfb_ms: float = 0.0  # streams: before the first byte, give or take a quarter
    chunks: int = 20  # streams: text deltas per reply
    reply_tokens: int = 200  # words of reply text, reported as output tokens
    error_share: float = 0.0  # replies that are an error, at once
    error_statuses: tuple = (429, 500, 529)
    slow_share: float = 0.0  # replies that take slow_ms instead of latency_ms
    slow_ms: float = 10_000.0
    agree_share: float = 0.9  # prompts on which the cheaper models give one letter
    differ_share: float = 0.02  # prompts on which the first cheaper model's letter is not the frontier's
    seed: int = 0
    cheap_models: tuple = CHEAP  # every other model answers as the frontier

    def argv(self) -> list[str]:
        out: list[str] = []
        for item in fields(self):
            value = getattr(self, item.name)
            out.append("--" + item.name.replace("_", "-"))
            out.extend(str(part) for part in (value if isinstance(value, tuple) else (value,)))
        return out


_ERRORS = {
    429: ("rate_limit_error", "rate_limit_exceeded", b"Too Many Requests"),
    500: ("api_error", "server_error", b"Internal Server Error"),
    529: ("overloaded_error", "server_error", b"Overloaded"),
}
_WORDS = ("the", "routing", "ledger", "answer", "model", "request", "check", "policy", "tenant", "cost", "frontier",
          "cheaper", "stage", "agreement", "shadow", "outcome", "latency", "token", "reply", "measured")
_CREATED = 1_763_337_600  # a fixed timestamp: replies must not depend on the clock


def answer_letter(settings: StandIn, of_prompt: bytes, model: str) -> str:
    """The letter a model gives for a prompt. The frontier's comes from the prompt's
    digest. On a share ``agree_share`` of prompts every cheaper model gives one letter,
    and on the rest each gives a different one. On a share ``differ_share`` the cheaper
    models' letters are shifted by one, so the first cheaper model's is not the
    frontier's."""
    frontier = of_prompt[24] % len(LETTERS)
    if model not in settings.cheap_models:
        return LETTERS[frontier]
    shift = int(_unit(of_prompt, 8) < settings.differ_share)
    if _unit(of_prompt, 0) >= settings.agree_share:
        shift += settings.cheap_models.index(model)
    return LETTERS[(frontier + shift) % len(LETTERS)]


def _sse(event: str, data: dict) -> bytes:
    return b"event: " + event.encode() + b"\ndata: " + json.dumps(data, separators=(",", ":")).encode() + b"\n\n"


def _framed(data: bytes) -> bytes:
    """One chunk of a chunked HTTP body."""
    return b"%x\r\n%s\r\n" % (len(data), data)


class StandInServer:
    """The stand-in's replies, and a small HTTP/1.1 server that sends them. It is a bare
    asyncio server so that it stays several times faster than the proxy it stands
    behind."""

    def __init__(self, settings: StandIn) -> None:
        self.settings = settings
        self.filler = " ".join(_WORDS[index % len(_WORDS)] for index in range(max(0, settings.reply_tokens)))
        self.requests = 0
        self.connections = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.streams = 0
        self.by_model: Counter = Counter()
        self.by_status: Counter = Counter()

    def plan(self, provider: str, raw: bytes) -> dict:
        """Everything about the reply to one request: status, how long it takes, the text."""
        settings = self.settings
        body = json.loads(raw)
        model = body.get("model") or ""
        prompt = json.dumps([body.get("system"), body["messages"]], sort_keys=True).encode()
        of_prompt = hashlib.sha256(b"%d|" % settings.seed + prompt).digest()
        of_call = hashlib.sha256(of_prompt + model.encode()).digest()
        slow = _unit(of_call, 16) < settings.slow_share
        status = 200
        if _unit(of_call, 8) < settings.error_share:
            status = settings.error_statuses[of_call[24] % len(settings.error_statuses)]
        options = body.get("stream_options")
        return {
            "status": status, "model": model, "stream": body.get("stream") is True, "jitter": _unit(of_call, 0),
            "seconds": ((settings.slow_ms if slow else settings.latency_ms)
                        + settings.latency_spread_ms * _unit(of_call, 0)) / 1000,
            "include_usage": isinstance(options, dict) and options.get("include_usage") is True,
            # From the prompt, not the request's bytes: a relay may write the same JSON differently.
            "id": of_call.hex()[:24], "input_tokens": max(1, len(prompt) // 4),
            "output_tokens": max(1, settings.reply_tokens),
            "text": f"Reply {of_call.hex()[:12]}. {self.filler}\nAnswer: {answer_letter(settings, of_prompt, model)}",
        }

    def payload(self, provider: str, plan: dict) -> dict:
        """A buffered reply as the provider's API sends it."""
        if provider == "anthropic":
            return {"id": f"msg_{plan['id']}", "type": "message", "role": "assistant", "model": plan["model"],
                    "content": [{"type": "text", "text": plan["text"]}], "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": plan["input_tokens"], "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 0, "output_tokens": plan["output_tokens"]}}
        return {"id": f"chatcmpl-{plan['id']}", "object": "chat.completion", "created": _CREATED,
                "model": plan["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": plan["text"], "refusal": None},
                             "logprobs": None, "finish_reason": "stop"}],
                "usage": self._openai_usage(plan)}

    @staticmethod
    def _openai_usage(plan: dict) -> dict:
        return {"prompt_tokens": plan["input_tokens"], "completion_tokens": plan["output_tokens"],
                "total_tokens": plan["input_tokens"] + plan["output_tokens"],
                "prompt_tokens_details": {"cached_tokens": 0}}

    def error(self, provider: str, status: int) -> dict:
        anthropic, openai, _ = _ERRORS.get(status, _ERRORS[500])
        message = f"stand-in provider: injected {status}"
        if provider == "anthropic":
            return {"type": "error", "error": {"type": anthropic, "message": message}}
        return {"error": {"message": message, "type": openai, "param": None, "code": openai}}

    def stream_parts(self, provider: str, plan: dict) -> tuple[bytes, list[bytes], bytes]:
        """A streamed reply as server-sent events: what opens it, one event per text
        delta, and what closes it. Anthropic carries input tokens in ``message_start``
        and output tokens in ``message_delta``; OpenAI sends a ``usage`` chunk only when
        the request asked with ``stream_options.include_usage``."""
        text = plan["text"]
        count = max(1, min(self.settings.chunks, len(text)))
        pieces = [text[len(text) * index // count:len(text) * (index + 1) // count] for index in range(count)]
        if provider == "anthropic":
            opening = _sse("message_start", {"type": "message_start", "message": {
                "id": f"msg_{plan['id']}", "type": "message", "role": "assistant", "model": plan["model"],
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": plan["input_tokens"], "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0, "output_tokens": 1}}})
            opening += _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                                    "content_block": {"type": "text", "text": ""}})
            opening += _sse("ping", {"type": "ping"})
            deltas = [_sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                   "delta": {"type": "text_delta", "text": piece}})
                      for piece in pieces]
            closing = _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            closing += _sse("message_delta", {"type": "message_delta",
                                              "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                              "usage": {"output_tokens": plan["output_tokens"]}})
            closing += _sse("message_stop", {"type": "message_stop"})
            return opening, deltas, closing
        base = {"id": f"chatcmpl-{plan['id']}", "object": "chat.completion.chunk", "created": _CREATED,
                "model": plan["model"]}
        # With include_usage every chunk carries a usage field: null until the last one.
        extra = {"usage": None} if plan["include_usage"] else {}

        def chunk(choices: list, **more) -> bytes:
            data = {**base, "choices": choices, **extra, **more}
            return b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n\n"
        opening = chunk([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
        deltas = [chunk([{"index": 0, "delta": {"content": piece}, "finish_reason": None}]) for piece in pieces]
        closing = chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}])
        if plan["include_usage"]:
            closing += chunk([], usage=self._openai_usage(plan))
        return opening, deltas, closing + b"data: [DONE]\n\n"

    def stats(self) -> dict:
        return {"requests": self.requests, "connections": self.connections, "in_flight": self.in_flight,
                "peak_in_flight": self.peak_in_flight, "streams": self.streams,
                "by_model": dict(self.by_model), "by_status": {str(key): value for key, value in self.by_status.items()}}

    @staticmethod
    def _json(status: int, data: dict, extra: bytes = b"") -> bytes:
        body = json.dumps(data, separators=(",", ":")).encode()
        reason = _ERRORS[status][2] if status in _ERRORS else {200: b"OK", 400: b"Bad Request", 404: b"Not Found"}[status]
        return (b"HTTP/1.1 %d %s\r\ncontent-type: application/json\r\n%scontent-length: %d\r\n\r\n"
                % (status, reason, extra, len(body))) + body

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        counted = False
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.split(b"\r\n")
                method, target, _ = lines[0].split(b" ", 2)
                if method == b"POST" and not counted:
                    # Connections that carried model requests: not this tool's own GETs.
                    counted = True
                    self.connections += 1
                length = 0
                for line in lines[1:]:
                    if line[:15].lower() == b"content-length:":
                        length = int(line[15:])
                raw = await reader.readexactly(length) if length else b""
                await self.respond(writer, method, target.split(b"?", 1)[0].decode("latin-1"), raw)
                await writer.drain()
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError, ValueError):
            pass  # the caller went away, or sent something that is not HTTP
        finally:
            writer.close()

    async def respond(self, writer: asyncio.StreamWriter, method: bytes, path: str, raw: bytes) -> None:
        if method == b"GET" and path in ("/health", "/stats"):
            writer.write(self._json(200, {"status": "ok", "standin": True} if path == "/health" else self.stats()))
            return
        provider = next((name for name, route in ROUTES.items() if route == path), None)
        if method != b"POST" or provider is None:
            writer.write(self._json(404, {"error": {"message": "the stand-in serves POST /v1/messages and POST "
                                                               "/v1/chat/completions", "type": "not_found"}}))
            return
        try:
            plan = self.plan(provider, raw)
        except (ValueError, TypeError, KeyError, AttributeError):
            writer.write(self._json(400, {"error": {"message": "the stand-in needs a JSON body with messages",
                                                    "type": "invalid_request_error"}}))
            return
        self.requests += 1
        self.by_model[plan["model"]] += 1
        self.by_status[plan["status"]] += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            request_id = b"request-id: req_%s\r\n" % plan["id"].encode()
            if plan["status"] != 200:
                retry = b"retry-after: 1\r\n" if plan["status"] == 429 else b""
                writer.write(self._json(plan["status"], self.error(provider, plan["status"]), request_id + retry))
            elif plan["stream"]:
                self.streams += 1
                await self.stream(writer, provider, plan, request_id)
            else:
                if plan["seconds"] > 0:
                    await asyncio.sleep(plan["seconds"])
                writer.write(self._json(200, self.payload(provider, plan), request_id))
        finally:
            self.in_flight -= 1

    async def stream(self, writer: asyncio.StreamWriter, provider: str, plan: dict, request_id: bytes) -> None:
        opening, deltas, closing = self.stream_parts(provider, plan)
        head = (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\n"
                b"%stransfer-encoding: chunked\r\n\r\n" % request_id)
        first = min(plan["seconds"], self.settings.ttfb_ms / 1000 * (0.75 + 0.5 * plan["jitter"]))
        gap = (plan["seconds"] - first) / len(deltas)
        if first > 0:
            await asyncio.sleep(first)
        if gap <= 0:
            # No pacing: still one HTTP chunk per event, so a relay handles each on its own.
            writer.write(head + _framed(opening) + b"".join(map(_framed, deltas)) + _framed(closing) + b"0\r\n\r\n")
            return
        writer.write(head + _framed(opening))
        await writer.drain()
        for delta in deltas[:-1]:
            await asyncio.sleep(gap)
            writer.write(_framed(delta))
            await writer.drain()
        await asyncio.sleep(gap)
        writer.write(_framed(deltas[-1]) + _framed(closing) + b"0\r\n\r\n")


async def serve_standin(settings: StandIn, host: str, port: int) -> None:
    """Serve until SIGTERM or SIGINT. Prints one JSON line when it is listening."""
    state = StandInServer(settings)
    server = await asyncio.start_server(state.handle, host, port, backlog=4096)
    stop = asyncio.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(name, stop.set)
    print(json.dumps({"standin": "listening", "host": host, "port": server.sockets[0].getsockname()[1],
                      "settings": asdict(settings)}), flush=True)
    await stop.wait()
    # Not ``wait_closed``: it would wait for every caller to hang up first.
    server.close()
    print(json.dumps({"standin": "stopped", **state.stats()}), flush=True)


# --- a small HTTP/1.1 client --------------------------------------------------------


class Conn:
    """One keep-alive connection: a request at a time."""
    __slots__ = ("reader", "writer", "fresh")

    def __init__(self) -> None:
        self.reader = self.writer = None
        self.fresh = True


@dataclass(slots=True)
class Reply:
    status: int
    headers: dict
    size: int
    first: float | None  # when the first byte of the body arrived (time.monotonic)
    sha: str | None
    body: bytes | None


class _Stale(Exception):
    """A reused connection that the server had already closed: nothing was answered."""


class Client:
    """Plain asyncio streams, so that the generator costs little CPU and its own
    overhead does not hide the proxy's. HTTP/1.1 with keep-alive, bodies by
    ``content-length`` or chunked."""

    def __init__(self, base_url: str) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise SystemExit(f"--base-url must be an http(s) URL, not {base_url!r}")
        self.host, self.tls = parts.hostname, parts.scheme == "https"
        self.port = parts.port or (443 if self.tls else 80)
        self.prefix = parts.path.rstrip("/")
        self.host_header = parts.netloc.encode()
        self.opened = 0
        self.reconnects = 0

    def close(self, conn: Conn) -> None:
        if conn.writer is not None:
            conn.writer.close()
        conn.reader = conn.writer = None

    def message(self, method: str, path: str, headers: bytes = b"", body: bytes | None = None) -> bytes:
        """The bytes of one request. ``headers`` are complete ``name: value\\r\\n`` lines."""
        head = b"%s %s HTTP/1.1\r\nhost: %s\r\n%s" % (method.encode(), (self.prefix + path).encode(),
                                                      self.host_header, headers)
        if body is None:
            return head + b"\r\n"
        return head + b"content-type: application/json\r\ncontent-length: %d\r\n\r\n" % len(body) + body

    async def send(self, conn: Conn, message: bytes, *, hashed: bool = False, keep: bool = False) -> Reply:
        """Send one request and read its whole reply. A reused connection that the
        server closed while it was idle is replaced once, as any HTTP client does."""
        for _ in range(2):
            if conn.writer is None:
                conn.reader, conn.writer = await asyncio.open_connection(
                    self.host, self.port, ssl=ssl.create_default_context() if self.tls else None)
                conn.fresh = True
                self.opened += 1
            reused, conn.fresh = not conn.fresh, False
            try:
                return await self._exchange(conn, message, hashed, keep, reused)
            except _Stale:
                self.close(conn)
                self.reconnects += 1
        raise ConnectionResetError("the server closed two connections in a row before answering")

    async def _exchange(self, conn: Conn, message: bytes, hashed: bool, keep: bool, reused: bool) -> Reply:
        reader, writer = conn.reader, conn.writer
        try:
            writer.write(message)
            await writer.drain()
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, ConnectionError) as error:
            if reused and not getattr(error, "partial", b""):
                raise _Stale from None
            raise
        status = int(head[9:12])
        headers = {}
        for line in head[:-4].split(b"\r\n")[1:]:
            name, _, value = line.partition(b":")
            headers[name.strip().lower()] = value.strip()
        digest = hashlib.sha256() if hashed else None
        parts: list[bytes] | None = [] if keep else None
        first, size = None, 0

        def took(data: bytes) -> None:
            nonlocal first, size
            if first is None:
                first = time.monotonic()
            size += len(data)
            if digest is not None:
                digest.update(data)
            if parts is not None:
                parts.append(data)
        if headers.get(b"transfer-encoding", b"").lower() == b"chunked":
            while True:
                length = int((await reader.readline()).split(b";", 1)[0], 16)
                if length == 0:
                    while (await reader.readline()) not in (b"\r\n", b"\n", b""):
                        pass  # trailers
                    break
                took((await reader.readexactly(length + 2))[:-2])
        elif b"content-length" in headers:
            length = int(headers[b"content-length"])
            if length:
                took(await reader.readexactly(length))
        elif status not in (204, 304):
            took(await reader.read())  # until the server closes
            self.close(conn)
        if headers.get(b"connection", b"").lower() == b"close":
            self.close(conn)
        return Reply(status, headers, size, first, digest.hexdigest() if digest else None,
                     b"".join(parts) if parts is not None else None)

    async def get(self, path: str, headers: bytes = b"", timeout: float = 60.0) -> Reply:
        """One GET on a connection of its own, with the body kept."""
        conn = Conn()
        try:
            async with asyncio.timeout(timeout):
                return await self.send(conn, self.message("GET", path, headers), keep=True)
        finally:
            self.close(conn)


def error_kind(error: BaseException) -> str:
    """A client-side failure, by kind: no HTTP reply came back for these."""
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, ConnectionRefusedError):
        return "connection_refused"
    if isinstance(error, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
        return "connection_reset"
    if isinstance(error, asyncio.IncompleteReadError):
        return "closed_before_reply_ended"
    if isinstance(error, (ValueError, asyncio.LimitOverrunError)):
        return "malformed_reply"
    if isinstance(error, OSError):
        return f"os_error_{error.errno}" if error.errno else type(error).__name__
    return type(error).__name__


# --- the load generator -------------------------------------------------------------


@dataclass
class Load:
    """One run of the generator. Give ``in_flight`` (closed loop: that many requests are
    always waiting for a reply) or ``rate`` (open loop: that many are sent each second
    whatever comes back), and ``duration`` or ``requests``."""
    base_url: str = ""
    in_flight: int | None = None
    rate: float | None = None
    duration: float | None = None  # seconds measured, after the warm-up
    requests: int | None = None  # requests measured, after the warm-up
    warmup_seconds: float = 0.0
    warmup_requests: int = 0
    stream_share: float = 0.0
    anthropic_share: float = 0.5  # the rest go to the OpenAI route
    prompt_chars: int = 2000
    max_tokens: int = 1024
    model: str = FRONTIER
    token: str = TOKEN
    outcome_share: float = 0.0  # of answered requests, reported as successes (PATCH /v1/outcomes)
    seed: int = 0
    nonce: str = ""  # part of every prompt: a new one makes every request new to a ledger or cache
    timeout: float = 120.0
    max_connections: int = 2000  # open loop only
    hash_bodies: bool = False

    def check(self) -> None:
        if (self.in_flight is None) == (self.rate is None):
            raise SystemExit("Give exactly one of --in-flight (closed loop) and --rate (open loop)")
        if self.duration is None and self.requests is None:
            raise SystemExit("Give --duration or --requests")
        if (self.in_flight or 1) < 1 or (self.rate or 1) <= 0 or (self.duration or 1) <= 0 or (self.requests or 1) < 1:
            raise SystemExit("--in-flight, --rate, --duration and --requests must be positive")
        for name in ("stream_share", "anthropic_share", "outcome_share"):
            if not 0 <= getattr(self, name) <= 1:
                raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")


class Requests:
    """The requests of one load. Route, streaming and prompt are decided from the seed,
    the nonce and the request's index alone, so a run at the stand-in and a run through
    the proxy send the same requests in the same order."""

    def __init__(self, load: Load, client: Client | None = None) -> None:
        self.load, self.client = load, client
        words = random.Random(load.seed)
        self.pool = " ".join(words.choice(_WORDS) for _ in range(load.prompt_chars // 3 + 400))
        self.heads = {
            "anthropic": b"x-api-key: %s\r\nanthropic-version: 2023-06-01\r\n" % load.token.encode(),
            "openai": b"authorization: Bearer %s\r\n" % load.token.encode()}

    def draws(self, index: int) -> bytes:
        return hashlib.blake2b(b"%d|%d" % (self.load.seed, index), digest_size=24).digest()

    def body(self, index: int) -> tuple[str, bool, dict]:
        """Route, whether it streams, and the JSON body of request ``index``."""
        load, draws = self.load, self.draws(index)
        route = "anthropic" if _unit(draws, 0) < load.anthropic_share else "openai"
        stream = _unit(draws, 8) < load.stream_share
        room = max(0, load.prompt_chars - 90)
        start = (index * 97) % max(1, len(self.pool) - room)
        prompt = (f"Load-test request {load.nonce}-{index}. {self.pool[start:start + room]}\n"
                  "End with a line 'Answer: X'.")
        body: dict = {"model": load.model, "messages": [{"role": "user", "content": prompt}]}
        if route == "anthropic":
            body["max_tokens"] = load.max_tokens
        if stream:
            body["stream"] = True
            if route == "openai":
                body["stream_options"] = {"include_usage": True}
        return route, stream, body

    def reports(self, index: int) -> bool:
        return _unit(self.draws(index), 16) < self.load.outcome_share

    def message(self, index: int) -> tuple[str, bool, bytes]:
        route, stream, body = self.body(index)
        return route, stream, self.client.message("POST", ROUTES[route], self.heads[route],
                                                  json.dumps(body, separators=(",", ":")).encode())


@dataclass(slots=True)
class Row:
    """One request as the generator saw it. Times are ``time.monotonic`` seconds."""
    index: int
    route: str
    stream: bool
    measured: bool  # not part of the warm-up
    scheduled: float
    sent: float
    first: float | None
    done: float
    status: int  # 0: no HTTP reply (see ``error``)
    error: str | None
    size: int
    sha: str | None
    via: str | None  # x-tine-route
    ledger: str | None  # x-tine-ledger
    cache: str | None  # x-tine-cache


async def run_load(load: Load) -> tuple[dict, list[Row]]:
    """Run one load. Returns its summary and every request's row."""
    load.check()
    client = Client(load.base_url)
    requests = Requests(load, client)
    rows: list[Row] = []
    outcomes: list[tuple[int, float]] = []
    patch_head = b"authorization: Bearer %s\r\n" % load.token.encode()
    loads_start = os.getloadavg()
    started = time.monotonic()
    measure_from = started + load.warmup_seconds
    total = None if load.requests is None else load.warmup_requests + load.requests
    stop_at = None if load.duration is None else measure_from + load.duration

    async def one(conn: Conn, index: int, scheduled: float) -> None:
        route, stream, message = requests.message(index)
        reply, error = None, None
        sent = time.monotonic()
        try:
            async with asyncio.timeout(load.timeout):
                reply = await client.send(conn, message, hashed=load.hash_bodies)
        except Exception as failure:
            error = error_kind(failure)
            client.close(conn)
        done = time.monotonic()
        headers = reply.headers if reply else {}

        def header(name: bytes) -> str | None:
            return headers[name].decode("latin-1") if name in headers else None
        rows.append(Row(index, route, stream, index >= load.warmup_requests and scheduled >= measure_from,
                        scheduled, sent, reply.first if reply else None, done, reply.status if reply else 0, error,
                        reply.size if reply else 0, reply.sha if reply else None, header(b"x-tine-route"),
                        header(b"x-tine-ledger"), header(b"x-tine-cache")))
        request_id = header(b"x-tine-request-id")
        if reply and reply.status == 200 and request_id and load.outcome_share and requests.reports(index):
            begun = time.monotonic()
            try:
                async with asyncio.timeout(load.timeout):
                    patched = await client.send(conn, client.message(
                        "PATCH", f"/v1/outcomes/{request_id}", patch_head, b'{"success":true}'))
                outcomes.append((patched.status, (time.monotonic() - begun) * 1000))
            except Exception:
                client.close(conn)
                outcomes.append((0, (time.monotonic() - begun) * 1000))

    if load.rate is None:
        taken = 0

        async def worker() -> None:
            nonlocal taken
            conn = Conn()
            while True:
                now = time.monotonic()
                if (total is not None and taken >= total) or (stop_at is not None and now >= stop_at):
                    break
                index, taken = taken, taken + 1
                await one(conn, index, now)
            client.close(conn)
        await asyncio.gather(*(worker() for _ in range(load.in_flight)))
    else:
        count = math.ceil(load.rate * (load.warmup_seconds + load.duration)) if load.duration is not None else total
        if total is not None:
            count = min(count, total)
        idle: list[Conn] = []
        gate = asyncio.Semaphore(load.max_connections)

        async def fire(index: int, scheduled: float) -> None:
            async with gate:
                conn = idle.pop() if idle else Conn()
                await one(conn, index, scheduled)
                idle.append(conn)
        tasks = []
        for index, offset in enumerate(schedule(load.rate, count)):
            # sleep(0) still lets the requests in flight run when the generator is behind.
            await asyncio.sleep(max(0.0, started + offset - time.monotonic()))
            tasks.append(asyncio.create_task(fire(index, started + offset)))
        await asyncio.gather(*tasks)
        for conn in idle:
            client.close(conn)
    ended = time.monotonic()
    summary = summarize(load, rows, measure_from=measure_from, stop_at=stop_at)
    summary.update({
        "seconds_total": round(ended - started, 3), "connections_opened": client.opened,
        "stale_connections_replaced": client.reconnects,
        "outcome_reports": {"sent": len(outcomes), "statuses": dict(Counter(str(status) for status, _ in outcomes)),
                            "latency_ms": distribution([ms for _, ms in outcomes])} if load.outcome_share else None,
        "load_average": {"start": [round(value, 2) for value in loads_start],
                         "end": [round(value, 2) for value in os.getloadavg()]},
        "cpu_count": os.cpu_count()})
    summary["busy_machine"] = busy(loads_start) or busy(os.getloadavg())
    summary["caveat"] = BUSY if summary["busy_machine"] else CAVEAT
    return summary, rows


def summarize(load: Load, rows: list[Row], *, measure_from: float, stop_at: float | None) -> dict:
    """Throughput, latency distribution, statuses and client-side errors of the measured
    requests. Latency figures are over replies with a 2xx status; others are listed
    apart, because a refusal that comes back at once is not a fast answer."""
    measured = [row for row in rows if row.measured]
    answered = [row for row in measured if row.status]
    good = [row for row in measured if 200 <= row.status < 300]
    if stop_at is not None:
        # A fixed window: every reply that arrived inside it, whenever it was asked for.
        window = stop_at - measure_from
        inside = [row for row in rows if row.status and measure_from <= row.done <= stop_at]
    else:
        window = max((row.done for row in measured), default=0.0) - min((row.scheduled for row in measured), default=0.0)
        inside = answered
    times = {id(row): timings(row.scheduled, row.sent, row.first, row.done) for row in measured}

    def of(chosen, name: str = "latency_ms") -> dict | None:
        return distribution([times[id(row)][name] for row in chosen if times[id(row)][name] is not None])
    streamed = [row for row in good if row.stream]

    def counted(name: str) -> dict:
        return dict(Counter(getattr(row, name) for row in measured if getattr(row, name) is not None))
    return {
        "target": load.base_url, "loop": "closed" if load.rate is None else "open",
        "in_flight": load.in_flight, "rate_rps": load.rate,
        "requests_sent": len(rows), "warmup_requests": len(rows) - len(measured), "measured_requests": len(measured),
        "window_seconds": round(window, 3),
        "throughput_rps": round(len(inside) / window, 2) if window > 0 else None,
        "ok_rps": round(sum(200 <= row.status < 300 for row in inside) / window, 2) if window > 0 else None,
        "statuses": dict(sorted(Counter(str(row.status) for row in answered).items())),
        "client_errors": dict(Counter(row.error for row in measured if row.error)),
        "latency_ms": of(good),
        "buffered_latency_ms": of([row for row in good if not row.stream]),
        "streamed_latency_ms": of(streamed),
        "streamed_ttfb_ms": of(streamed, "ttfb_ms"),
        "not_2xx_latency_ms": of([row for row in answered if not 200 <= row.status < 300]),
        # Open loop: how late requests left, and the time after they left. A send delay
        # that grows means the target (or the generator) could not keep up with the rate.
        "send_delay_ms": of(measured, "send_delay_ms") if load.rate is not None else None,
        "service_ms": of(good, "service_ms") if load.rate is not None else None,
        "by_route": {route: of([row for row in good if row.route == route]) for route in ROUTES},
        "bytes_received": sum(row.size for row in measured),
        "proxy_headers": {"x-tine-route": counted("via"), "x-tine-ledger": counted("ledger"),
                          "x-tine-cache": counted("cache")},
    }


def windows(rows: list[Row], seconds: float) -> list[dict]:
    """Latency per stretch of ``seconds``, by scheduled send time: a soak's slow growth
    shows as a drift from the first stretch to the last."""
    measured = [row for row in rows if row.measured]
    if not measured:
        return []
    begin = min(row.scheduled for row in measured)
    slots: dict[int, list[Row]] = {}
    for row in measured:
        slots.setdefault(int((row.scheduled - begin) // seconds), []).append(row)
    out = []
    for slot in sorted(slots):
        good = [(row.done - row.scheduled) * 1000 for row in slots[slot] if 200 <= row.status < 300]
        out.append({"from_seconds": round(slot * seconds, 1), "requests": len(slots[slot]),
                    "not_2xx": sum(row.status != 0 and not 200 <= row.status < 300 for row in slots[slot]),
                    "client_errors": sum(row.status == 0 for row in slots[slot]), "latency_ms": distribution(good)})
    return out


def growth(samples: list[dict], stretches: list[dict]) -> dict | None:
    """What changed between the first and the last sample of a run: the slow growth a
    soak is for. Memory that keeps rising per request, or a median that drifts from the
    first stretch to the last, is the thing to look at."""
    known = [sample for sample in samples if sample.get("rss_bytes") is not None and sample.get("ledger_rows") is not None]
    if len(known) < 2:
        return None
    first, last = known[0], known[-1]
    served = last["ledger_rows"] - first["ledger_rows"]
    medians = [stretch["latency_ms"]["p50"] for stretch in stretches if stretch["latency_ms"]]
    return {
        "from_seconds": first["seconds"], "to_seconds": last["seconds"], "requests_between": served,
        "rss_bytes_first": first["rss_bytes"], "rss_bytes_last": last["rss_bytes"],
        "rss_bytes_per_request": round((last["rss_bytes"] - first["rss_bytes"]) / served, 1) if served else None,
        "ledger_bytes_per_request": round((last["ledger_bytes"] - first["ledger_bytes"]) / served, 1) if served else None,
        "p50_ms_first_stretch": medians[0] if medians else None, "p50_ms_last_stretch": medians[-1] if medians else None,
    }


def write_rows(path: Path, rows: list[Row]) -> None:
    """Per-request rows as JSON lines, times in milliseconds from the first send."""
    begin = min((row.scheduled for row in rows), default=0.0)
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps({
                "index": row.index, "route": row.route, "stream": row.stream, "measured": row.measured,
                "scheduled_ms": round((row.scheduled - begin) * 1000, 3), "status": row.status, "error": row.error,
                **{key: None if value is None else round(value, 3)
                   for key, value in timings(row.scheduled, row.sent, row.first, row.done).items()},
                "bytes": row.size, "sha256": row.sha, "x_tine_route": row.via, "x_tine_ledger": row.ledger,
                "x_tine_cache": row.cache}) + "\n")


# --- processes ----------------------------------------------------------------------

_CHILDREN: list["Child"] = []
_next_port = PORTS[0] + (os.getpid() * 37) % 800


def free_port() -> int:
    """A port in 18000-18999 that nothing is listening on."""
    global _next_port
    for _ in range(PORTS[1] - PORTS[0] + 1):
        port, _next_port = _next_port, _next_port + 1 if _next_port < PORTS[1] else PORTS[0]
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"No free port between {PORTS[0]} and {PORTS[1]}")


def child_environment(settings: dict | None = None) -> dict:
    """The environment of a server this tool starts: no provider keys, none of the
    caller's Tine settings, and no HTTP proxy, because every call it makes is to the
    stand-in on this machine."""
    dropped = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in dropped and (not key.startswith("TINE_") or key == "TINE_DEV_DIR")}
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONUNBUFFERED"] = "1"
    env["NO_PROXY"] = env["no_proxy"] = "*"
    env.update(settings or {})
    return env


class Child:
    """A server process. Its exit is collected with ``wait4``, which also gives its peak
    memory and CPU time; ``subprocess``'s own waiting would throw those away."""

    def __init__(self, argv: list[str], *, env: dict, log: Path) -> None:
        self.log = log
        with log.open("wb") as out:
            self.process = subprocess.Popen(argv, env=env, cwd=REPO, stdout=out, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL)
        self.pid = self.process.pid
        self.exit_code: int | None = None
        self.usage = None
        _CHILDREN.append(self)

    def exited(self, flags: int = os.WNOHANG) -> bool:
        if self.exit_code is not None:
            return True
        try:
            pid, status, usage = os.wait4(self.pid, flags)
        except ChildProcessError:
            self.exit_code = -1
            return True
        if pid == 0:
            return False
        self.exit_code, self.usage = os.waitstatus_to_exitcode(status), usage
        self.process.returncode = self.exit_code
        return True

    def stop(self, grace: float = 20.0) -> dict:
        """Ask the process to stop, wait, and return what it used."""
        deadline = time.monotonic() + grace
        killed = False
        if not self.exited():
            os.kill(self.pid, signal.SIGTERM)
        while not self.exited():
            if time.monotonic() > deadline and not killed:
                os.kill(self.pid, signal.SIGKILL)
                killed = True
            time.sleep(0.02)
        usage = self.usage
        # ru_maxrss is bytes on macOS and kilobytes on Linux.
        scale = 1 if sys.platform == "darwin" else 1024
        return {"exit_code": self.exit_code, "killed_after_grace": killed,
                "peak_rss_bytes": usage.ru_maxrss * scale if usage else None,
                "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 3) if usage else None,
                "log_bytes": self.log.stat().st_size if self.log.exists() else None}

    def log_tail(self, lines: int = 15) -> str:
        try:
            return "\n".join(self.log.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""


@atexit.register
def _stop_children() -> None:
    for child in _CHILDREN:
        if not child.exited():
            try:
                os.kill(child.pid, signal.SIGKILL)
            except OSError:
                pass


def process_now(pid: int) -> dict | None:
    """A running process's resident memory and the CPU time it has used so far, read
    from the system for that one process: libproc on macOS, /proc on Linux. None where
    neither is there."""
    try:
        if sys.platform == "darwin":
            usage = (ctypes.c_uint64 * 32)()  # rusage_info_v0: twelve 64-bit fields
            if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:
                return None
            # User and system time are in Mach time units, not nanoseconds.
            base = (ctypes.c_uint32 * 2)()
            ctypes.CDLL("/usr/lib/libSystem.B.dylib").mach_timebase_info(ctypes.byref(base))
            return {"rss_bytes": int(usage[8]),
                    "cpu_seconds": (usage[2] + usage[3]) * base[0] / base[1] / 1e9}
        with open(f"/proc/{pid}/statm") as statm:
            rss = int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        with open(f"/proc/{pid}/stat") as stat:
            after_name = stat.read().rsplit(")", 1)[1].split()
        return {"rss_bytes": rss, "cpu_seconds": (int(after_name[11]) + int(after_name[12])) / os.sysconf("SC_CLK_TCK")}
    except (OSError, ValueError, IndexError, AttributeError, ZeroDivisionError):
        return None


def raise_file_limit() -> int:
    """Hundreds of requests in flight need hundreds of sockets in each process, and a
    shell's usual limit is 256. Children inherit what is set here."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    for wanted in (65536, 10240, 4096):
        target = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
        if soft >= target:
            break
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            break
        except (ValueError, OSError):
            continue
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


async def wait_ready(url: str, child: Child | None, seconds: float, what: str) -> None:
    client = Client(url)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if child is not None and child.exited():
            raise RuntimeError(f"{what} exited with code {child.exit_code} before it answered:\n{child.log_tail()}")
        try:
            if (await client.get("/health", timeout=2.0)).status == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(0.1)
    raise RuntimeError(f"{what} at {url} did not answer GET /health within {seconds:.0f} s"
                       + (f":\n{child.log_tail()}" if child else ""))


async def start_standin(settings: StandIn, workdir: Path, *, host: str = "127.0.0.1",
                        port: int | None = None) -> tuple[Child, str]:
    port = port or free_port()
    child = Child([sys.executable, str(Path(__file__).resolve()), "provider", "--host", host, "--port", str(port),
                   *settings.argv()], env=child_environment(), log=workdir / f"standin-{port}.log")
    url = f"http://127.0.0.1:{port}"
    await wait_ready(url, child, 30.0, "The stand-in provider")
    return child, url


def synthetic_policy() -> dict:
    """A policy the proxy accepts, for load only. Its validator requires a certified
    guarantee on any policy with stages, so one is written; nothing was certified."""
    return {
        "version": 1, "tenant": TENANT, "task_type": TASK_TYPE, "frontier": FRONTIER,
        "stages": [{"models": list(CHEAP), "check": "agreement"}],
        "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.04},
        "answer_pattern": ANSWER_PATTERN, "synthetic": True,
        "note": ("SYNTHETIC LOAD-TEST POLICY written by scripts/load_test.py. Nothing was certified: the guarantee "
                 "fields are placeholders that the proxy's validator requires, the models are a stand-in "
                 "provider's, and no number produced with this file is evidence about quality or savings."),
    }


@dataclass
class Scenario:
    """One thing to measure: a stand-in, a load and how the proxy is configured."""
    name: str
    about: str = ""
    provider: dict = field(default_factory=dict)  # StandIn fields
    load: dict = field(default_factory=dict)  # Load fields
    mode: str = "passthrough"  # or "shadow", "route"
    sample_rate: float = 0.0  # shadow mode
    audit_rate: float = 0.0  # routing: the frontier holdout
    timeout_seconds: float | None = None  # TINE_TIMEOUT_SECONDS; None leaves the proxy's default (30)
    rounds: int = 3
    direct_seconds: float | None = None  # measure the stand-in for this long only (the soak)
    sample_seconds: float | None = None  # sample memory and ledger size this often during proxy runs
    window_seconds: float | None = None  # report latency per stretch of this length
    kind: str = "compare"  # or "ledger"
    sizes: tuple = ()  # kind "ledger": rows to fill before measuring
    environment: dict = field(default_factory=dict)  # further TINE_ settings

    def seconds(self) -> float:
        """Roughly how long it takes to run."""
        if self.kind == "ledger":
            return sum(25 + size / 2500 for size in self.sizes)
        each = self.load.get("warmup_seconds", 0.0) + self.load.get("duration", 0.0)
        direct = self.load.get("warmup_seconds", 0.0) + self.direct_seconds if self.direct_seconds else each
        return 12 + self.rounds * (each + direct + (3 if self.mode == "shadow" else 0))


def proxy_settings(scenario: Scenario, provider_url: str, policies: str | None) -> dict:
    """The proxy's environment for a scenario. ``policies`` is the policy file's path as
    the proxy sees it."""
    settings = {
        "TINE_TENANT_KEYS": json.dumps({TOKEN: TENANT}),
        "TINE_ANTHROPIC_API_KEY": FAKE_KEY, "TINE_OPENAI_API_KEY": FAKE_KEY,
        "TINE_ANTHROPIC_FRONTIER_MODEL": FRONTIER, "TINE_OPENAI_FRONTIER_MODEL": FRONTIER,
        "TINE_ANTHROPIC_BASE_URL": provider_url, "TINE_OPENAI_BASE_URL": provider_url,
        "TINE_MODEL_PRICES": json.dumps(PRICES),
    }
    if scenario.timeout_seconds is not None:
        settings["TINE_TIMEOUT_SECONDS"] = str(scenario.timeout_seconds)
    if scenario.mode == "shadow":
        # The cap is in made-up dollars at made-up prices: high enough never to stop a run.
        settings.update({"TINE_SHADOW_POLICIES": policies, "TINE_SHADOW_SAMPLE_RATE": str(scenario.sample_rate),
                         "TINE_SHADOW_MAX_COST_USD": "1000000"})
    elif scenario.mode == "route":
        settings.update({"TINE_ROUTE_POLICIES": policies, "TINE_AUDIT_RATE": str(scenario.audit_rate),
                         "TINE_ROUTE_ESTIMATE_BASELINE": "true"})
    elif scenario.mode != "passthrough":
        raise SystemExit(f"Unknown mode {scenario.mode!r}: passthrough, shadow or route")
    settings.update(scenario.environment)
    return settings


def write_policies(folder: Path) -> Path:
    """The synthetic policy for both routes, in a folder a container can read."""
    folder.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o755)
    path = folder / "load-test-policies.json"
    path.write_text(json.dumps({TENANT: {"anthropic": synthetic_policy(), "openai": synthetic_policy()}}, indent=2))
    path.chmod(0o644)
    return path


def docker_line(scenario: Scenario, *, provider_port: int, base_url: str, policies: Path | None,
                image: str = "tine-proxy:dev", name: str = "tine-load-proxy") -> str:
    """The exact ``docker run`` line for a proxy container that this scenario can be
    measured through. The container reaches the stand-in on the host as
    ``host.docker.internal``; settings are passed with ``-e``, never an env file."""
    published = urlsplit(base_url).port or 80
    inside = f"/config/{policies.name}" if policies else None
    parts = ["docker", "run", "--rm", "-d", "--name", name, "-p", f"127.0.0.1:{published}:8000",
             "--add-host", "host.docker.internal:host-gateway"]
    if policies is not None and scenario.mode != "passthrough":
        parts += ["-v", f"{policies.parent}:/config:ro"]
    for key, value in proxy_settings(scenario, f"http://host.docker.internal:{provider_port}", inside).items():
        parts += ["-e", f"{key}={value}"]
    return shlex.join([*parts, image])


def ledger_stats(path: Path | None) -> dict | None:
    """Size on disk (with the write-ahead log) and row counts, read from a second,
    read-only connection."""
    if path is None or not path.exists():
        return None
    size = sum(part.stat().st_size for part in (path, Path(f"{path}-wal")) if part.exists())
    rows: dict = {}
    try:
        db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=10)
        try:
            for table in ("requests", "shadow_runs", "outcome_events"):
                try:
                    rows[table] = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except sqlite3.OperationalError:
                    rows[table] = None
        finally:
            db.close()
    except sqlite3.Error as error:
        rows["error"] = type(error).__name__
    return {"bytes": size, "rows": rows,
            "bytes_per_request_row": round(size / rows["requests"]) if rows.get("requests") else None}


async def timed_get(client: Client, path: str, times: int = 3) -> dict:
    """How long a report endpoint takes, as the tenant, a few times over."""
    head = b"authorization: Bearer %s\r\n" % TOKEN.encode()
    took, reply = [], None
    for _ in range(times):
        begun = time.monotonic()
        try:
            reply = await client.get(path, head, timeout=300.0)
        except Exception as error:
            return {"status": 0, "error": error_kind(error), "ms": [round(value, 2) for value in took]}
        took.append((time.monotonic() - begun) * 1000)
    try:
        body = json.loads(reply.body)
    except ValueError:
        body = None
    return {"status": reply.status, "median_ms": round(statistics.median(took), 2),
            "ms": [round(value, 2) for value in took], "bytes": reply.size, "body": body}


_SHADOW_FIELDS = (
    "sampling", "sampled_requests", "shadowed_requests", "shadow_coverage", "skipped_busy", "skipped_stopped",
    "failed_shadow_runs", "skipped_rate_limited", "skipped_rate_cap", "rate_limited_runs", "streamed_not_shadowed",
    "policy_would_accept", "policy_would_accept_share", "accepted_compared_with_frontier",
    "accepted_matching_frontier_share", "shadow_calls", "failed_shadow_calls", "unknown_cost_shadow_calls",
    "drift_check")


async def standin_stats(provider_url: str) -> dict:
    return json.loads((await Client(provider_url).get("/stats")).body)


async def drain(provider_url: str, limit: float = 60.0) -> float | None:
    """Seconds until the stand-in has had nothing in flight for a quarter of a second:
    how long the proxy's background work (shadow runs) went on after the last reply.
    None if it never went quiet."""
    begun = time.monotonic()
    quiet_since = None
    while time.monotonic() - begun < limit:
        if (await standin_stats(provider_url))["in_flight"] == 0:
            quiet_since = quiet_since or time.monotonic()
            if time.monotonic() - quiet_since >= 0.25:
                return round(quiet_since - begun, 3)
        else:
            quiet_since = None
        await asyncio.sleep(0.02)
    return None


def same_bytes(direct: list[Row], through: list[Row]) -> dict:
    """Whether the proxy relayed the provider's bytes: bodies of the same request, asked
    directly and through the proxy, compared by SHA-256 where both got a 200."""
    theirs = {row.index: row for row in direct if row.status == 200 and row.sha}
    out = {"streamed": Counter(), "buffered": Counter()}
    for row in through:
        other = theirs.get(row.index)
        if row.status == 200 and row.sha and other is not None and other.stream == row.stream:
            out["streamed" if row.stream else "buffered"]["equal" if row.sha == other.sha else "different"] += 1
    return {kind: {"compared": sum(counts.values()), "equal": counts["equal"], "different": counts["different"]}
            for kind, counts in out.items()}


def machine() -> dict:
    return {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "open_file_limit": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
            "load_average": [round(value, 2) for value in os.getloadavg()]}


# --- compare ------------------------------------------------------------------------


async def compare(scenario: Scenario, workdir: Path, *, base_url: str | None = None, provider_host: str = "127.0.0.1",
                  provider_port: int | None = None, ledger: Path | None = None, wait_seconds: float = 180.0,
                  image: str = "tine-proxy:dev") -> dict:
    """Measure one scenario at the stand-in directly and through the proxy.

    Without ``base_url`` the proxy is ``python -m tine serve`` started here with a fresh
    ledger in ``workdir``. With it, the proxy is one that someone else starts (a
    container): the ``docker run`` line it needs is printed and this waits for it."""
    settings = StandIn(**{key: tuple(value) if isinstance(value, list) else value
                          for key, value in scenario.provider.items()})
    policies = write_policies(workdir / "config") if scenario.mode != "passthrough" else None
    standin, provider_url = await start_standin(settings, workdir, host=provider_host, port=provider_port)
    proxy, docker = None, None
    try:
        if base_url is None:
            port = free_port()
            ledger = workdir / "ledger.sqlite3"
            proxy = Child([sys.executable, "-m", "tine", "serve", "--host", "127.0.0.1", "--port", str(port)],
                          env=child_environment({**proxy_settings(scenario, provider_url, str(policies)),
                                                 "TINE_LEDGER_PATH": str(ledger)}),
                          log=workdir / "proxy.log")
            proxy_url = f"http://127.0.0.1:{port}"
        else:
            proxy_url = base_url.rstrip("/")
            docker = docker_line(scenario, provider_port=urlsplit(provider_url).port, base_url=proxy_url,
                                 policies=policies, image=image)
            say("Start the proxy container with exactly this line (the stand-in is already listening):\n\n  "
                + docker + "\n\nOn Linux the stand-in must also listen where the container can reach it: add "
                "--provider-host 0.0.0.0 to this command. Remove the container afterwards with: "
                "docker rm -f tine-load-proxy\n")
        await wait_ready(proxy_url, proxy, 90.0 if proxy else wait_seconds, "The proxy")
        # A proxy started by someone else must be the one this scenario describes.
        health = json.loads((await Client(proxy_url).get("/health")).body)
        wanted = {"routing": "policy" if scenario.mode == "route" else "frontier-default",
                  "shadow": "sampling" if scenario.mode == "shadow" and scenario.sample_rate > 0 else "off"}
        wrong = {key: health[key] for key, value in wanted.items() if key in health and health[key] != value}
        if wrong:
            raise RuntimeError(f"The proxy at {proxy_url} reports {wrong} on GET /health, but scenario "
                               f"{scenario.name!r} needs {wanted}: it was not started with this scenario's settings")
        return await _measure(scenario, settings, workdir, provider_url, proxy_url, proxy, ledger, docker)
    finally:
        if proxy is not None:
            await asyncio.to_thread(proxy.stop)
        await asyncio.to_thread(standin.stop)


async def _measure(scenario: Scenario, settings: StandIn, workdir: Path, provider_url: str, proxy_url: str,
                   proxy: Child | None, ledger: Path | None, docker: str | None) -> dict:
    began = machine()
    run = f"{int(time.time())}-{os.getpid()}"
    tenant = Client(proxy_url)
    rounds, pooled = [], {"direct": [], "proxy": []}
    samples: list[dict] = []
    drains: list[float | None] = []
    equal = {"streamed": Counter(), "buffered": Counter()}
    upstream: Counter = Counter()
    # CPU time used before the first measured request (starting up) is not the load's.
    cpu_before = (process_now(proxy.pid) if proxy else None) or {}

    async def sampler(number: int) -> None:
        # The proxy's memory and CPU time and the ledger's size, during a run through it.
        begun = time.monotonic()
        while True:
            await asyncio.sleep(scenario.sample_seconds)
            stats = ledger_stats(ledger) or {}
            now = (process_now(proxy.pid) if proxy else None) or {}
            samples.append({"round": number, "seconds": round(time.monotonic() - begun, 1),
                            "rss_bytes": now.get("rss_bytes"),
                            "cpu_seconds": round(now["cpu_seconds"], 3) if now else None,
                            "ledger_bytes": stats.get("bytes"), "ledger_rows": (stats.get("rows") or {}).get("requests")})

    for number in range(scenario.rounds):
        # The same requests on both sides, and the order swapped every round, so that a
        # machine that gets busier or quieter during the run weighs on both.
        order = ("direct", "proxy") if number % 2 == 0 else ("proxy", "direct")
        entry: dict = {"round": number, "order": list(order)}
        rows: dict[str, list[Row]] = {}
        for side in order:
            load = Load(**{**scenario.load, "base_url": provider_url if side == "direct" else proxy_url,
                           "nonce": f"{run}-r{number}", "hash_bodies": True})
            if side == "direct":
                # The stand-in knows no outcome endpoint; a soak measures it only briefly.
                load = replace(load, outcome_share=0.0, duration=scenario.direct_seconds or load.duration)
            watching = (asyncio.create_task(sampler(number))
                        if side == "proxy" and scenario.sample_seconds and proxy is not None else None)
            before = await standin_stats(provider_url)
            try:
                entry[side], rows[side] = await run_load(load)
            finally:
                if watching is not None:
                    watching.cancel()
            pooled[side].extend(rows[side])
            if side == "proxy":
                if scenario.mode == "shadow":
                    drains.append(await drain(provider_url))
                after = await standin_stats(provider_url)
                # What the proxy asked of the stand-in, and over how many connections.
                upstream.update({"requests": after["requests"] - before["requests"],
                                 "connections": after["connections"] - before["connections"]})
        for kind, counts in same_bytes(rows["direct"], rows["proxy"]).items():
            equal[kind].update({"equal": counts["equal"], "different": counts["different"]})
        rounds.append(entry)
        say(f"  {scenario.name} round {number + 1}/{scenario.rounds}: direct p50 "
            f"{(entry['direct']['latency_ms'] or {}).get('p50')} ms at {entry['direct']['throughput_rps']}/s, proxy p50 "
            f"{(entry['proxy']['latency_ms'] or {}).get('p50')} ms at {entry['proxy']['throughput_rps']}/s")

    def per_round(side: str, group: str, name: str) -> list:
        return [(entry[side][group] or {}).get(name) for entry in rounds]

    def added(group: str, name: str) -> dict | None:
        return spread([None if through is None or direct is None else through - direct
                       for through, direct in zip(per_round("proxy", group, name), per_round("direct", group, name))])

    def together(side: str) -> dict:
        measured = [row for row in pooled[side] if row.measured]
        good = [row for row in measured if 200 <= row.status < 300]
        streamed = [row for row in good if row.stream]
        return {"measured_requests": len(measured),
                "statuses": dict(sorted(Counter(str(row.status) for row in measured if row.status).items())),
                "client_errors": dict(Counter(row.error for row in measured if row.error)),
                "throughput_rps": spread([entry[side]["throughput_rps"] for entry in rounds]),
                "latency_ms": distribution([(row.done - row.scheduled) * 1000 for row in good]),
                "streamed_ttfb_ms": distribution([(row.first - row.scheduled) * 1000 for row in streamed if row.first])}
    ratio = spread([None if not direct or through is None else through / direct for through, direct in
                    zip([entry["proxy"]["throughput_rps"] for entry in rounds],
                        [entry["direct"]["throughput_rps"] for entry in rounds])])
    ledger_now = ledger_stats(ledger)
    savings = await timed_get(tenant, "/v1/savings")
    shadow = await timed_get(tenant, "/v1/shadow")
    provider_stats = await standin_stats(provider_url)
    cpu_after = (process_now(proxy.pid) if proxy else None) or {}
    # The proxy stops here so that its peak memory and CPU time can be read.
    process = await asyncio.to_thread(proxy.stop) if proxy is not None else None
    served = sum(entry["proxy"]["requests_sent"] for entry in rounds)
    if process is not None:
        process["rss_bytes_before_load"] = cpu_before.get("rss_bytes")
        process["rss_bytes_after_load"] = cpu_after.get("rss_bytes")
        process["requests_served"] = served
        # Everything the proxy did for the load (outcome reports and shadow runs too),
        # per request the generator sent. One process has one core: 1000 over this
        # figure is the most requests a second it could serve.
        used = cpu_after["cpu_seconds"] - cpu_before["cpu_seconds"] if cpu_before and cpu_after else None
        process["cpu_ms_per_request"] = round(used * 1000 / served, 3) if served and used is not None else None
        # Near 1.0 the proxy's one core was the limit; well below it at full load, the
        # proxy was waiting for something else (the disk, the stand-in, a busy machine).
        running = sum(entry["proxy"]["seconds_total"] for entry in rounds)
        process["cpu_share_of_one_core_during_runs"] = round(used / running, 3) if running and used is not None else None
    result = {
        "scenario": scenario.name, "about": scenario.about, "caveat": CAVEAT,
        "busy_machine": any(entry[side]["busy_machine"] for entry in rounds for side in ("direct", "proxy")),
        "machine": {**began, "load_average_end": [round(value, 2) for value in os.getloadavg()]},
        "settings": {"standin": asdict(settings), "load": scenario.load, "mode": scenario.mode,
                     "sample_rate": scenario.sample_rate, "audit_rate": scenario.audit_rate,
                     "proxy_timeout_seconds": scenario.timeout_seconds or "the proxy's default (30)",
                     "rounds": scenario.rounds,
                     "proxy": "python -m tine serve, started by this tool" if proxy else f"already running at {proxy_url}"},
        "docker_run": docker,
        "direct": together("direct"), "proxy": together("proxy"),
        # Through the proxy minus direct, per round: the median over rounds and their range.
        "added_latency_ms": {name: added("latency_ms", name) for name in ("p50", "p95", "p99")},
        "added_ttfb_ms": {name: added("streamed_ttfb_ms", name) for name in ("p50", "p95")},
        "throughput_ratio": ratio,
        "relayed_bytes": {kind: {"compared": sum(counts.values()), **dict(counts)} for kind, counts in equal.items()},
        "proxy_process": process,
        "ledger": ledger_now,
        "report_endpoints": {
            "savings": {key: value for key, value in savings.items() if key != "body"},
            "shadow": {key: value for key, value in shadow.items() if key != "body"}},
        "shadow": None, "routing": None,
        # What the proxy sent upstream during its runs. Many more connections than the
        # most requests in flight means they are opened and closed instead of reused,
        # which costs a TLS handshake each against a real provider.
        "upstream": {"requests": upstream["requests"], "connections_opened": upstream["connections"]},
        "standin": provider_stats,
        "rounds": rounds,
    }
    if scenario.mode == "shadow" and isinstance(shadow.get("body"), dict):
        result["shadow"] = {**{key: shadow["body"].get(key) for key in _SHADOW_FIELDS},
                            "seconds_shadow_calls_ran_on_after_each_round": drains,
                            "standin_calls_by_model": provider_stats["by_model"]}
    if scenario.mode == "route":
        routes: Counter = Counter()
        for entry in rounds:
            routes.update(entry["proxy"]["proxy_headers"]["x-tine-route"])
        holdout = (savings.get("body") or {}).get("holdout_estimate") or {}
        result["routing"] = {"x_tine_route": dict(routes),
                             "holdout_estimate": {key: value for key, value in holdout.items()
                                                  if key not in ("method", "scope")} or None,
                             "standin_calls_by_model": provider_stats["by_model"]}
    if scenario.window_seconds:
        result["windows"] = windows(pooled["proxy"], scenario.window_seconds)
    if scenario.sample_seconds:
        result["samples"] = samples
        result["growth"] = growth(samples, result.get("windows") or [])
    result["observations"] = observations(result)
    if result["busy_machine"]:
        result["caveat"] = BUSY
    return result


def brief(result: dict) -> dict:
    """A result as it is printed: each round cut down to its headline figures. The file
    written with ``--output`` keeps every round's whole summary."""
    def short(summary: dict) -> dict:
        latency = summary["latency_ms"] or {}
        return {"throughput_rps": summary["throughput_rps"], "measured_requests": summary["measured_requests"],
                "latency_ms": {name: latency.get(name) for name in ("p50", "p95", "p99", "max")},
                "statuses": summary["statuses"], "client_errors": summary["client_errors"],
                "load_average_start": summary["load_average"]["start"][0]}
    return {**result, "rounds": [{**entry, "direct": short(entry["direct"]), "proxy": short(entry["proxy"])}
                                 for entry in result.get("rounds", [])]}


def observations(result: dict) -> list[str]:
    """What a reader should look at first: where the proxy answered differently from the
    provider, and where the measurement itself is in doubt."""
    out = []
    direct, proxy = result["direct"], result["proxy"]
    if result["busy_machine"]:
        out.append(BUSY)
    for status, count in proxy["statuses"].items():
        other = direct["statuses"].get(status, 0)
        if status != "200" and count > 2 * other + 2:
            out.append(f"The proxy answered {status} to {count} of {proxy['measured_requests']} measured requests; "
                       f"the stand-in answered {status} to {other} of {direct['measured_requests']} asked directly")
    good = [side["statuses"].get("200", 0) / side["measured_requests"] if side["measured_requests"] else 0.0
            for side in (direct, proxy)]
    if abs(good[0] - good[1]) > 0.02:
        out.append(f"{good[1]:.1%} of requests through the proxy got a 200 against {good[0]:.1%} asked directly, so "
                   "the latency percentiles on the two sides are over different sets of requests and their "
                   "difference is not the proxy's added latency")
    for side, name in ((proxy, "through the proxy"), (direct, "at the stand-in")):
        if side["client_errors"]:
            out.append(f"Requests {name} that got no HTTP reply: {side['client_errors']}")
    unrecorded = sum(entry["proxy"]["proxy_headers"]["x-tine-ledger"].get("unavailable", 0) for entry in result["rounds"])
    if unrecorded:
        out.append(f"{unrecorded} replies carried x-tine-ledger: unavailable: the proxy could not write their ledger row")
    ratio = result["throughput_ratio"]
    closed = all(entry["proxy"]["loop"] == "closed" for entry in result["rounds"])
    if closed and ratio and ratio["median"] > 0.7 and not result["settings"]["standin"]["latency_ms"]:
        out.append("With instant replies the proxy kept more than 0.7 of the direct throughput: the generator or the "
                   "stand-in may be the limit here, not the proxy")
    for entry in result["rounds"]:
        delay = entry["proxy"].get("send_delay_ms")
        if delay and delay["p99"] > 20:
            out.append(f"Round {entry['round']}: requests left up to {delay['max']} ms late (p99 {delay['p99']} ms): "
                       "the proxy or the generator could not keep up with the open-loop rate")
            break
    for kind, counts in result["relayed_bytes"].items():
        if counts.get("different") and result["settings"]["mode"] != "route":
            out.append(f"{counts['different']} of {counts['compared']} {kind} replies through the proxy were not "
                       "byte for byte the stand-in's")
    process = result["proxy_process"]
    # uvicorn shuts down on SIGTERM and then raises the signal again: -15 is a clean stop.
    if process and (process["exit_code"] not in (0, -signal.SIGTERM) or process["killed_after_grace"]):
        out.append(f"The proxy process ended with exit code {process['exit_code']}"
                   + (" after it had to be killed: it did not stop on SIGTERM" if process["killed_after_grace"] else ""))
    upstream = result["upstream"]
    if upstream["connections_opened"] > max(200, 0.2 * upstream["requests"]):
        out.append(f"The proxy opened {upstream['connections_opened']} connections to the stand-in for "
                   f"{upstream['requests']} upstream requests. Its HTTP client keeps 20 idle connections by default, "
                   "and httpcore closes a connection that goes idle whenever more than 20 are open, so above 20 "
                   "upstream requests in flight nearly every request opens a new one: a TLS handshake each against a "
                   "real provider, which this loopback measurement does not include")
    shadow = result.get("shadow")
    if shadow and shadow.get("sampled_requests"):
        out.append(f"Shadow mode ran {shadow['shadowed_requests']} of {shadow['sampled_requests']} sampled requests "
                   f"(coverage {shadow['shadow_coverage']}); skipped_busy {shadow['skipped_busy']}, "
                   f"failed {shadow['failed_shadow_runs']}; sampling is {shadow['sampling']!r}")
    for name in ("savings", "shadow"):
        report = result["report_endpoints"][name]
        if report.get("median_ms", 0) > 250:
            out.append(f"GET /v1/{name} took {report['median_ms']} ms, and the proxy answers nothing else meanwhile")
    return out


# --- ledger growth ------------------------------------------------------------------


def fill_ledger(path: Path, size: int, *, prompt_chars: int, reply_tokens: int) -> dict:
    """Write ``size`` request rows and as many shadow rows straight into a ledger with
    ``tine.ledger.Ledger.record`` and ``tine.shadow.ShadowStore.record``: the proxy's
    own writers, without the proxy. The rows look like the ones a pass-through request
    and its shadow run leave."""
    sys.path.insert(0, str(REPO))
    from tine.ledger import Ledger
    from tine.shadow import ShadowStore

    begun = time.monotonic()
    ledger = Ledger(path)
    shadow = ShadowStore(ledger)
    standin = StandInServer(StandIn(reply_tokens=reply_tokens))
    requests = Requests(Load(prompt_chars=prompt_chars, nonce="fill"))
    policy = synthetic_policy()
    for index in range(size):
        route, _, body = requests.body(index)
        plan = standin.plan(route, json.dumps(body).encode())
        cost = (plan["input_tokens"] * 10.0 + plan["output_tokens"] * 50.0) / 1e6
        request_id = ledger.record(
            tenant=TENANT, provider=route, model=FRONTIER, requested_model=FRONTIER, request_body=body,
            response_body=standin.payload(route, plan), upstream_status=200, input_tokens=plan["input_tokens"],
            output_tokens=plan["output_tokens"], cached_tokens=0, cost_usd=cost, baseline_cost_usd=cost,
            latency_ms=2.0, success=True if index % 5 == 0 else None)
        accepted = index % 10 != 0
        shadow.record(
            request_id=request_id, tenant=TENANT, provider=route, policy=policy, frontier_answer="A",
            frontier_cost_usd=cost,
            outcome={"accepted": accepted, "stage": 0 if accepted else None, "served_by": CHEAP[0] if accepted else None,
                     "answer": "A" if accepted else None, "checks": [],
                     "calls": [{"model": model, "cost_usd": cost / 10, "error": None} for model in CHEAP]})
    ledger.close()
    seconds = time.monotonic() - begun
    return {"rows": size, "seconds": round(seconds, 2), "rows_per_second": round(size / seconds) if seconds else None,
            "filled_by": "tine.ledger.Ledger.record and tine.shadow.ShadowStore.record directly, not through the proxy"}


async def ledger_growth(scenario: Scenario, workdir: Path) -> dict:
    """For each size: fill a fresh ledger, start the proxy on it with shadow sampling on,
    and time requests one at a time, the two report endpoints, and a request sent while
    a report is being made (the reports are computed on the proxy's one event loop)."""
    settings = StandIn(**scenario.provider)
    policies = write_policies(workdir / "config")
    shadowing = replace(scenario, mode="shadow", sample_rate=1.0)
    out = []
    for size in scenario.sizes:
        folder = workdir / f"ledger-{size}"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        path = folder / "ledger.sqlite3"
        say(f"  {scenario.name}: filling {size:,} rows")
        filled = await asyncio.to_thread(fill_ledger, path, size, prompt_chars=scenario.load.get("prompt_chars", 2000),
                                         reply_tokens=settings.reply_tokens)
        standin, provider_url = await start_standin(settings, folder)
        port = free_port()
        proxy = Child([sys.executable, "-m", "tine", "serve", "--host", "127.0.0.1", "--port", str(port)],
                      env=child_environment({**proxy_settings(shadowing, provider_url, str(policies)),
                                             "TINE_LEDGER_PATH": str(path)}), log=folder / "proxy.log")
        proxy_url = f"http://127.0.0.1:{port}"
        try:
            begun = time.monotonic()
            await wait_ready(proxy_url, proxy, 120.0, "The proxy")
            startup = time.monotonic() - begun
            tenant = Client(proxy_url)
            asked = scenario.load.get("requests", 40)
            load = Load(base_url=proxy_url, in_flight=1, requests=asked, nonce=f"g{size}-{int(time.time())}",
                        prompt_chars=scenario.load.get("prompt_chars", 2000))
            summary, rows = await run_load(load)
            one_by_one = [round((row.done - row.sent) * 1000, 2) for row in rows]
            await drain(provider_url)
            savings = await timed_get(tenant, "/v1/savings")
            shadow = await timed_get(tenant, "/v1/shadow")
            # A request that arrives just after a report was asked for.
            behind = []
            for attempt in range(3):
                report = asyncio.create_task(tenant.get("/v1/savings", b"authorization: Bearer %s\r\n" % TOKEN.encode(),
                                                        timeout=300.0))
                await asyncio.sleep(0.005)
                _, waited = await run_load(replace(load, requests=1, nonce=f"{load.nonce}-b{attempt}"))
                behind.append(round((waited[0].done - waited[0].sent) * 1000, 2))
                await report
            stats = ledger_stats(path)
        finally:
            process = await asyncio.to_thread(proxy.stop)
            await asyncio.to_thread(standin.stop)
        slowest = max(range(len(one_by_one)), key=one_by_one.__getitem__) if one_by_one else None
        entry = {
            "rows_filled": size, "fill": filled, "proxy_startup_seconds": round(startup, 2),
            "requests_one_at_a_time": {"latency_ms": summary["latency_ms"], "statuses": summary["statuses"],
                                       "first_ms": one_by_one[0] if one_by_one else None,
                                       "second_ms": one_by_one[1] if len(one_by_one) > 1 else None,
                                       "slowest_ms": one_by_one[slowest] if one_by_one else None,
                                       "slowest_was_request_number": slowest},
            "savings": {key: value for key, value in savings.items() if key != "body"},
            "shadow": {key: value for key, value in shadow.items() if key != "body"},
            "request_ms_while_savings_report_runs": behind,
            "ledger": stats, "proxy_process": process, "busy_machine": summary["busy_machine"],
        }
        out.append(entry)
        say(f"  {scenario.name} at {size:,} rows: request p50 {(summary['latency_ms'] or {}).get('p50')} ms, "
            f"/v1/savings {savings.get('median_ms')} ms, /v1/shadow {shadow.get('median_ms')} ms")
        shutil.rmtree(folder, ignore_errors=True)
    notes = []
    for entry in out:
        if entry["busy_machine"] and BUSY not in notes:
            notes.append(BUSY)
        timed = entry["requests_one_at_a_time"]
        median = (timed["latency_ms"] or {}).get("p50")
        waited = max(entry["request_ms_while_savings_report_runs"], default=0.0)
        if median and waited > 5 * median + 10:
            notes.append(f"At {entry['rows_filled']:,} rows GET /v1/savings took {entry['savings'].get('median_ms')} ms "
                         f"and GET /v1/shadow {entry['shadow'].get('median_ms')} ms; a request sent while the savings "
                         f"report was being made took {waited} ms against a median of {median} ms. The reports read "
                         "the tenant's whole table on the proxy's one event loop, so nothing else is answered meanwhile")
        if median and timed["slowest_ms"] > 10 * median + 20:
            notes.append(f"At {entry['rows_filled']:,} rows request number {timed['slowest_was_request_number']} (from "
                         f"0) took {timed['slowest_ms']} ms against a median of {median} ms. The first shadow run "
                         "after a start reads every stored shadow row of the tenant, also on the event loop")
    return {"scenario": scenario.name, "about": scenario.about, "caveat": CAVEAT, "machine": machine(),
            "busy_machine": any(entry["busy_machine"] for entry in out),
            "settings": {"standin": asdict(settings), "shadow_sample_rate": 1.0, "sizes": list(scenario.sizes)},
            "sizes": out, "observations": notes}


# --- the matrix ---------------------------------------------------------------------


def scenarios(*, quick: bool = False, soak_seconds: float = 300.0, soak_rate: float = 20.0,
              ledger_sizes: tuple = (10_000, 100_000)) -> list[Scenario]:
    """The named scenarios. ``quick`` shrinks every one to a few seconds (shorter
    replies, one round, short runs): a smoke run that proves the tool works, not a
    measurement."""
    scale = 0.2 if quick else 1.0
    rounds = 1 if quick else 3
    warm = 0.5 if quick else 2.0

    def ms(value: float) -> float:
        return value * scale

    def lasting(full: float, slowest_reply_ms: float = 0.0) -> float:
        return max(2.0, 3 * ms(slowest_reply_ms) / 1000) if quick else full
    instant = {"latency_ms": 0.0}
    real = {"latency_ms": ms(1500), "latency_spread_ms": ms(1000)}
    medium = {"latency_ms": ms(300), "latency_spread_ms": ms(300)}
    out = [Scenario(
        f"a-passthrough-instant-{n}", f"Pass-through, instant replies, {n} in flight: the proxy's own overhead"
        + (" and its ceiling" if n >= 50 else ""), provider=instant, rounds=rounds,
        load={"in_flight": n, "duration": lasting(10), "warmup_seconds": warm}) for n in (1, 10, 50, 200)]
    out += [Scenario(
        f"b-passthrough-realistic-{n}", f"Pass-through, replies of about 2 s (1.5 to 2.5), {n} in flight",
        provider=real, rounds=rounds, load={"in_flight": n, "duration": lasting(seconds, 2500), "warmup_seconds": warm})
        for n, seconds in ((10, 40), (50, 30))]
    out += [
        Scenario("c-streams-instant", "Streams with no provider delay, 10 in flight: what relaying a stream costs",
                 provider={**instant, "chunks": 20}, rounds=rounds,
                 load={"in_flight": 10, "duration": lasting(10), "warmup_seconds": warm, "stream_share": 1.0}),
        Scenario("c-streams-realistic", "Streams of about 2 s in 150 events with the first byte after about 0.4 s, "
                 "20 in flight: time to first byte and total", provider={**real, "ttfb_ms": ms(400), "chunks": 150},
                 rounds=rounds,
                 load={"in_flight": 20, "duration": lasting(30, 2500), "warmup_seconds": warm, "stream_share": 1.0}),
    ]
    policy_load = {"in_flight": 20, "duration": lasting(20, 600), "warmup_seconds": warm}
    out += [Scenario(
        f"d-shadow-{rate}", "Shadow mode with the synthetic agreement policy at sample rate " + str(rate)
        + (" (the policy loaded, nothing sampled: the baseline for the two below)" if not rate else "")
        + ", replies of 0.3 to 0.6 s, 20 in flight", provider=medium, rounds=rounds, load=policy_load,
        mode="shadow", sample_rate=rate) for rate in (0.0, 0.1, 1.0)]
    out += [Scenario(
        f"e-route-agree-{share}", f"Live routing through the synthetic agreement policy: the cheaper models agree on "
        f"{share:.0%} of prompts, 10% of requests held out for the frontier, replies of 0.3 to 0.6 s, 20 in flight",
        provider={**medium, "agree_share": share}, rounds=rounds, load=policy_load, mode="route", audit_rate=0.1)
        for share in (0.9, 0.3)]
    out.append(Scenario(
        "f-errors-and-slow", "Pass-through with 10% provider errors (429, 500, 529) and 10% replies of 6 s against a "
        "proxy timeout of 4 s, other replies 0.3 to 0.6 s, 20 in flight",
        provider={**medium, "error_share": 0.1, "slow_share": 0.1, "slow_ms": ms(6000)}, rounds=rounds,
        timeout_seconds=ms(4000) / 1000,
        load={"in_flight": 20, "duration": lasting(25, 6600), "warmup_seconds": warm}))
    out.append(Scenario(
        "g-ledger-growth", "A request, GET /v1/savings and GET /v1/shadow on a ledger already holding "
        + " and ".join(f"{size:,}" for size in ledger_sizes) + " rows (filled directly through tine.ledger.Ledger, "
        "not through the proxy)", provider=instant, kind="ledger", sizes=tuple(ledger_sizes),
        load={"requests": 20 if quick else 60}))
    out.append(Scenario(
        "h-soak", f"A soak: {soak_seconds:.0f} s through the proxy at a steady {soak_rate:g} requests a second (open "
        "loop), half of them reporting an outcome, with latency per stretch and memory and ledger size sampled",
        provider={"latency_ms": ms(200), "latency_spread_ms": ms(200)}, rounds=1,
        direct_seconds=min(30.0, soak_seconds), sample_seconds=max(1.0, soak_seconds / 30),
        window_seconds=max(2.0, soak_seconds / 5) if soak_seconds < 300 else 60.0,
        load={"rate": soak_rate, "duration": soak_seconds, "warmup_seconds": warm, "outcome_share": 0.5}))
    out += [
        Scenario("i-pool-queue", "250 requests in flight with replies of about 2 s: more than the 100 connections "
                 "the proxy's upstream pool allows by default, with the default 30 s timeout",
                 provider=real, rounds=1, load={"in_flight": 250, "duration": lasting(30, 2500), "warmup_seconds": warm}),
        Scenario("i-pool-timeout", "400 requests in flight with replies of 1.5 to 2 s and the proxy's timeout set to "
                 "3 s, so that the wait for a pooled connection runs out (at the default 30 s the same takes longer)",
                 provider={"latency_ms": ms(1500), "latency_spread_ms": ms(500)}, rounds=1,
                 timeout_seconds=ms(3000) / 1000,
                 load={"in_flight": 400, "duration": lasting(25, 4000), "warmup_seconds": warm}),
    ]
    return out


CSV_COLUMNS = (
    "scenario", "mode", "in_flight", "rate_rps", "stream_share", "ledger_rows_filled", "rounds", "busy_machine",
    "direct_rps", "proxy_rps", "throughput_ratio", "direct_p50_ms", "proxy_p50_ms", "direct_p99_ms", "proxy_p99_ms",
    "proxy_max_ms", "added_p50_ms", "added_p50_min_ms", "added_p50_max_ms", "added_p95_ms", "added_p99_ms",
    "added_ttfb_p50_ms", "added_ttfb_p95_ms", "proxy_statuses", "proxy_client_errors", "relayed_bytes_different",
    "proxy_peak_rss_mb", "proxy_cpu_ms_per_request", "proxy_cpu_share", "upstream_requests", "upstream_connections",
    "ledger_rows", "ledger_bytes", "savings_ms", "shadow_ms",
    "request_ms_while_report_runs", "shadow_coverage", "shadow_skipped_busy", "x_tine_route", "load_average_start",
    "load_average_end", "observations")


def csv_rows(result: dict) -> list[dict]:
    """A scenario's result as rows of the combined table: one, or one per ledger size."""
    def compact(counts: dict | None) -> str:
        return " ".join(f"{key}:{value}" for key, value in sorted((counts or {}).items()))

    def megabytes(process: dict | None):
        return round(process["peak_rss_bytes"] / 2**20, 1) if process and process.get("peak_rss_bytes") else None
    if "error" in result:
        return [{"scenario": result["scenario"], "observations": f"FAILED: {result['error']}"}]
    if "sizes" in result:
        return [{
            "scenario": f"{result['scenario']}-{entry['rows_filled']}", "mode": "shadow", "in_flight": 1,
            "ledger_rows_filled": entry["rows_filled"], "busy_machine": entry["busy_machine"],
            "proxy_p50_ms": (entry["requests_one_at_a_time"]["latency_ms"] or {}).get("p50"),
            "proxy_p99_ms": (entry["requests_one_at_a_time"]["latency_ms"] or {}).get("p99"),
            "proxy_max_ms": entry["requests_one_at_a_time"]["slowest_ms"],
            "proxy_statuses": compact(entry["requests_one_at_a_time"]["statuses"]),
            "proxy_peak_rss_mb": megabytes(entry["proxy_process"]),
            "ledger_rows": (entry["ledger"] or {}).get("rows", {}).get("requests"),
            "ledger_bytes": (entry["ledger"] or {}).get("bytes"),
            "savings_ms": entry["savings"].get("median_ms"), "shadow_ms": entry["shadow"].get("median_ms"),
            "request_ms_while_report_runs": max(entry["request_ms_while_savings_report_runs"], default=None),
            "load_average_start": result["machine"]["load_average"][0],
            "observations": len(result["observations"])} for entry in result["sizes"]]
    direct, proxy, load = result["direct"], result["proxy"], result["settings"]["load"]

    def middle(value: dict | None, name: str = "median"):
        return value[name] if value else None
    added, first = result["added_latency_ms"], result["added_ttfb_ms"]
    return [{
        "scenario": result["scenario"], "mode": result["settings"]["mode"], "in_flight": load.get("in_flight"),
        "rate_rps": load.get("rate"), "stream_share": load.get("stream_share", 0.0),
        "rounds": result["settings"]["rounds"], "busy_machine": result["busy_machine"],
        "direct_rps": middle(direct["throughput_rps"]), "proxy_rps": middle(proxy["throughput_rps"]),
        "throughput_ratio": middle(result["throughput_ratio"]),
        "direct_p50_ms": middle(direct["latency_ms"], "p50"), "proxy_p50_ms": middle(proxy["latency_ms"], "p50"),
        "direct_p99_ms": middle(direct["latency_ms"], "p99"), "proxy_p99_ms": middle(proxy["latency_ms"], "p99"),
        "proxy_max_ms": middle(proxy["latency_ms"], "max"),
        "added_p50_ms": middle(added["p50"]), "added_p50_min_ms": middle(added["p50"], "min"),
        "added_p50_max_ms": middle(added["p50"], "max"), "added_p95_ms": middle(added["p95"]),
        "added_p99_ms": middle(added["p99"]), "added_ttfb_p50_ms": middle(first["p50"]),
        "added_ttfb_p95_ms": middle(first["p95"]), "proxy_statuses": compact(proxy["statuses"]),
        "proxy_client_errors": compact(proxy["client_errors"]),
        # Under routing a cheaper model's reply is expected to differ from the frontier's.
        "relayed_bytes_different": (None if result["settings"]["mode"] == "route" else
                                    sum(counts.get("different", 0) for counts in result["relayed_bytes"].values())),
        "proxy_peak_rss_mb": megabytes(result["proxy_process"]),
        "proxy_cpu_ms_per_request": (result["proxy_process"] or {}).get("cpu_ms_per_request"),
        "proxy_cpu_share": (result["proxy_process"] or {}).get("cpu_share_of_one_core_during_runs"),
        "upstream_requests": result["upstream"]["requests"],
        "upstream_connections": result["upstream"]["connections_opened"],
        "ledger_rows": ((result["ledger"] or {}).get("rows") or {}).get("requests"),
        "ledger_bytes": (result["ledger"] or {}).get("bytes"),
        "savings_ms": result["report_endpoints"]["savings"].get("median_ms"),
        "shadow_ms": result["report_endpoints"]["shadow"].get("median_ms"),
        "shadow_coverage": (result["shadow"] or {}).get("shadow_coverage"),
        "shadow_skipped_busy": (result["shadow"] or {}).get("skipped_busy"),
        "x_tine_route": compact((result["routing"] or {}).get("x_tine_route")),
        "load_average_start": result["machine"]["load_average"][0],
        "load_average_end": result["machine"]["load_average_end"][0],
        "observations": len(result["observations"])}]


async def ceiling(seconds: float, workdir: Path, in_flight: int = 32) -> dict:
    """How fast the stand-in answers when it answers at once, buffered and streamed,
    asked by this generator. The proxy must be well below this for a comparison to be
    about the proxy."""
    child, url = await start_standin(StandIn(), workdir)
    try:
        out = {}
        for name, share in (("buffered", 0.0), ("streamed", 1.0)):
            summary, _ = await run_load(Load(base_url=url, in_flight=in_flight, duration=seconds, warmup_seconds=0.3,
                                             stream_share=share, nonce=f"ceiling-{name}"))
            out[name] = {"requests_per_second": summary["throughput_rps"], "latency_ms": summary["latency_ms"],
                         "statuses": summary["statuses"], "client_errors": summary["client_errors"]}
        loads = os.getloadavg()
    finally:
        await asyncio.to_thread(child.stop)
    return {"standin_ceiling": out, "in_flight": in_flight, "seconds_each": seconds,
            "what": "the stand-in and this generator together, each one process, replies at once",
            "load_average": [round(value, 2) for value in loads], "busy_machine": busy(loads),
            "caveat": BUSY if busy(loads) else CAVEAT}


async def matrix(chosen: list[Scenario], output: Path, workdir: Path, *, quick: bool) -> dict:
    """Run scenarios in order. Writes ``<name>.json`` for each, ``standin-ceiling.json``,
    the combined ``matrix.csv`` and ``matrix.json`` into ``output``. Every scenario gets
    its own stand-in, proxy process and empty ledger; one that fails is recorded and the
    rest still run."""
    output.mkdir(parents=True, exist_ok=True)
    table: list[dict] = []
    files = {}
    begun = time.monotonic()
    top = await ceiling(2.0 if quick else 5.0, workdir)
    (output / "standin-ceiling.json").write_text(json.dumps(top, indent=2) + "\n")
    say(f"stand-in ceiling: {top['standin_ceiling']['buffered']['requests_per_second']} buffered and "
        f"{top['standin_ceiling']['streamed']['requests_per_second']} streamed replies a second")
    for number, scenario in enumerate(chosen, 1):
        say(f"[{number}/{len(chosen)}] {scenario.name}: {scenario.about}")
        folder = workdir / scenario.name
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        try:
            result = await (ledger_growth(scenario, folder) if scenario.kind == "ledger" else compare(scenario, folder))
        except Exception as error:  # one scenario that fails must not lose the others
            result = {"scenario": scenario.name, "about": scenario.about, "error": f"{type(error).__name__}: {error}"}
            say(f"  FAILED: {result['error']}")
        result["quick"] = quick
        files[scenario.name] = f"{scenario.name}.json"
        (output / files[scenario.name]).write_text(json.dumps(result, indent=2) + "\n")
        table.extend(csv_rows(result))
        for note in result.get("observations", []):
            say(f"  note: {note}")
        shutil.rmtree(folder, ignore_errors=True)
    with (output / "matrix.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    summary = {"scenarios": files, "table": "matrix.csv", "standin_ceiling": "standin-ceiling.json", "quick": quick,
               "seconds": round(time.monotonic() - begun, 1), "machine": machine(),
               "busy_machine": top["busy_machine"] or any(row.get("busy_machine") for row in table),
               "failed": [row["scenario"] for row in table if str(row.get("observations", "")).startswith("FAILED")],
               "caveat": CAVEAT}
    if quick:
        summary["caveat"] = "A quick smoke run: it proves the tool works and measures nothing. " + CAVEAT
    elif summary["busy_machine"]:
        summary["caveat"] = BUSY
    (output / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# --- command line -------------------------------------------------------------------


def add_standin_arguments(parser) -> None:
    group = parser.add_argument_group("the stand-in provider")
    group.add_argument("--latency-ms", type=float, default=0.0, help="every reply takes this long (default: at once)")
    group.add_argument("--latency-spread-ms", type=float, default=0.0,
                       help="plus up to this much, different for every request")
    group.add_argument("--ttfb-ms", type=float, default=0.0, help="streams: time to the first byte, give or take 25%%")
    group.add_argument("--chunks", type=int, default=20, help="streams: text deltas per reply")
    group.add_argument("--reply-tokens", type=int, default=200, help="length of the reply text")
    group.add_argument("--error-share", type=float, default=0.0, help="share of replies that are an error")
    group.add_argument("--error-statuses", type=int, nargs="+", default=[429, 500, 529])
    group.add_argument("--slow-share", type=float, default=0.0, help="share of replies that take --slow-ms")
    group.add_argument("--slow-ms", type=float, default=10_000.0)
    group.add_argument("--agree-share", type=float, default=0.9,
                       help="share of prompts on which the cheaper models give the same letter")
    group.add_argument("--differ-share", type=float, default=0.02,
                       help="share of prompts on which the first cheaper model's letter is not the frontier's")
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--cheap-models", nargs="+", default=list(CHEAP),
                       help="models that answer as cheaper models; every other model answers as the frontier")


def standin_from(args) -> StandIn:
    values = {item.name: getattr(args, item.name) for item in fields(StandIn)}
    return StandIn(**{key: tuple(value) if isinstance(value, list) else value for key, value in values.items()})


def add_load_arguments(parser) -> None:
    group = parser.add_argument_group("the load")
    group.add_argument("--in-flight", type=int, help="closed loop: keep this many requests waiting for a reply")
    group.add_argument("--rate", type=float, help="open loop: send this many requests a second, whatever comes back; "
                                                  "latency is counted from the scheduled send")
    group.add_argument("--duration", type=float, help="seconds to measure, after the warm-up")
    group.add_argument("--requests", type=int, help="requests to measure, after the warm-up")
    group.add_argument("--warmup-seconds", type=float, default=0.0, help="left out of every figure")
    group.add_argument("--warmup-requests", type=int, default=0, help="left out of every figure")
    group.add_argument("--stream-share", type=float, default=0.0, help="share of requests that stream")
    group.add_argument("--anthropic-share", type=float, default=0.5,
                       help="share sent to POST /v1/messages; the rest go to POST /v1/chat/completions")
    group.add_argument("--prompt-chars", type=int, default=2000, help="size of each (distinct) prompt")
    group.add_argument("--max-tokens", type=int, default=1024)
    group.add_argument("--model", default=FRONTIER)
    group.add_argument("--token", default=TOKEN, help="the proxy tenant token (a made-up one by default)")
    group.add_argument("--outcome-share", type=float, default=0.0,
                       help="share of answered requests reported as a success with PATCH /v1/outcomes")
    group.add_argument("--load-seed", dest="load_seed", type=int, default=0)
    group.add_argument("--timeout", type=float, default=120.0, help="seconds the generator waits for one reply")
    group.add_argument("--max-connections", type=int, default=2000, help="open loop: connections at most")


def load_from(args, **more) -> dict:
    names = ("in_flight", "rate", "duration", "requests", "warmup_seconds", "warmup_requests", "stream_share",
             "anthropic_share", "prompt_chars", "max_tokens", "model", "token", "outcome_share", "timeout",
             "max_connections")
    return {**{name: getattr(args, name) for name in names}, "seed": args.load_seed, **more}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog="No real model is called and nothing leaves this machine.")
    commands = parser.add_subparsers(dest="command", required=True)

    provider = commands.add_parser("provider", help="Run the stand-in provider")
    provider.add_argument("--host", default="127.0.0.1")
    provider.add_argument("--port", type=int, default=PORTS[0] + 1)
    provider.add_argument("--ceiling", type=float, metavar="SECONDS",
                          help="measure how fast the stand-in is (buffered, then streamed), print it and exit")
    add_standin_arguments(provider)

    run = commands.add_parser("run", help="Send a load to a base URL and summarise it")
    run.add_argument("--base-url", required=True, help="the stand-in, a local 'python -m tine serve' or a container")
    run.add_argument("--nonce", default=None, help="part of every prompt (default: the time, so every run is new)")
    run.add_argument("--rows", type=Path, help="also write one JSON line per request to this local scratch file")
    run.add_argument("--output", type=Path, help="also write the summary here")
    add_load_arguments(run)

    both = commands.add_parser("compare", help="Measure one scenario at the stand-in and through the proxy")
    both.add_argument("--name", default="compare")
    both.add_argument("--mode", choices=("passthrough", "shadow", "route"), default="passthrough")
    both.add_argument("--sample-rate", type=float, default=1.0, help="shadow mode: TINE_SHADOW_SAMPLE_RATE")
    both.add_argument("--audit-rate", type=float, default=0.0, help="routing: TINE_AUDIT_RATE, the frontier holdout")
    both.add_argument("--proxy-timeout", type=float, help="TINE_TIMEOUT_SECONDS (the proxy's default is 30)")
    both.add_argument("--rounds", type=int, default=3, help="direct and proxied runs alternate this many times")
    both.add_argument("--base-url", help="measure a proxy that is already running (a container) instead of starting "
                                         "one; the docker run line it must be started with is printed")
    both.add_argument("--image", default="tine-proxy:dev", help="the image named in the printed docker run line")
    both.add_argument("--provider-host", default="127.0.0.1", help="where the stand-in listens")
    both.add_argument("--provider-port", type=int, help="the stand-in's port (default: a free one in 18000-18999)")
    both.add_argument("--wait-seconds", type=float, default=180.0, help="with --base-url: wait this long for the proxy")
    both.add_argument("--ledger", type=Path, help="with --base-url: the proxy's ledger file, if this machine can read it")
    both.add_argument("--sample-seconds", type=float, help="sample the proxy's memory and ledger size this often")
    both.add_argument("--window-seconds", type=float, help="also report latency per stretch of this many seconds")
    both.add_argument("--output", type=Path, help="also write the result here")
    both.add_argument("--workdir", type=Path, help="where the temporary ledger and logs go (default: a temp folder)")
    both.add_argument("--keep", action="store_true", help="keep that folder (it holds synthetic prompts and logs)")
    add_standin_arguments(both)
    add_load_arguments(both)

    table = commands.add_parser("matrix", help="Run the named scenarios, one JSON each and a combined CSV")
    table.add_argument("--output", type=Path, help="folder for the results (for example artifacts/load-test/DATE)")
    table.add_argument("--list", action="store_true", help="list the scenarios and how long they take, and exit")
    table.add_argument("--only", nargs="+", metavar="NAME", help="run only scenarios whose name starts with one of these")
    table.add_argument("--quick", action="store_true",
                       help="a smoke run of a few seconds per scenario: proves the tool works, measures nothing")
    table.add_argument("--soak-seconds", type=float, help="length of the soak (default 300; 15 with --quick)")
    table.add_argument("--soak-rate", type=float, default=20.0, help="requests a second in the soak")
    table.add_argument("--ledger-sizes", type=int, nargs="+", help="rows to fill (default 10000 100000; "
                                                                   "500 3000 with --quick)")
    table.add_argument("--workdir", type=Path)
    table.add_argument("--keep", action="store_true")

    args = parser.parse_args(argv)
    raise_file_limit()

    if args.command == "provider":
        if args.ceiling:
            with tempfile.TemporaryDirectory(prefix="tine-load-") as folder:
                print(json.dumps(asyncio.run(ceiling(args.ceiling, Path(folder))), indent=2))
            return
        asyncio.run(serve_standin(standin_from(args), args.host, args.port))
        return

    if args.command == "run":
        load = Load(**load_from(args, base_url=args.base_url, hash_bodies=bool(args.rows),
                                nonce=args.nonce if args.nonce is not None else f"{int(time.time())}-{os.getpid()}"))
        summary, rows = asyncio.run(run_load(load))
        if args.rows:
            write_rows(args.rows, rows)
            summary["rows_file"] = str(args.rows)
        text = json.dumps(summary, indent=2)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        return

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="tine-load-"))
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "compare":
            scenario = Scenario(
                args.name, "One scenario from the command line", provider=asdict(standin_from(args)),
                load=load_from(args), mode=args.mode, sample_rate=args.sample_rate if args.mode == "shadow" else 0.0,
                audit_rate=args.audit_rate, timeout_seconds=args.proxy_timeout, rounds=args.rounds,
                sample_seconds=args.sample_seconds, window_seconds=args.window_seconds)
            Load(**{**scenario.load, "base_url": "http://check"}).check()
            result = asyncio.run(compare(
                scenario, workdir, base_url=args.base_url, provider_host=args.provider_host,
                provider_port=args.provider_port, ledger=args.ledger, wait_seconds=args.wait_seconds, image=args.image))
            if args.output:
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(brief(result), indent=2))
            return
        quick = args.quick
        chosen = scenarios(
            quick=quick, soak_seconds=args.soak_seconds or (15.0 if quick else 300.0), soak_rate=args.soak_rate,
            ledger_sizes=tuple(args.ledger_sizes or ((500, 3000) if quick else (10_000, 100_000))))
        if args.only:
            chosen = [scenario for scenario in chosen if scenario.name.startswith(tuple(args.only))]
            if not chosen:
                raise SystemExit("--only matched no scenario; see matrix --list")
        if args.list:
            for scenario in chosen:
                print(f"{scenario.name:32} ~{scenario.seconds():5.0f} s  {scenario.about}")
            print(f"{'total':32} ~{sum(scenario.seconds() for scenario in chosen) / 60:5.1f} min")
            return
        if args.output is None:
            raise SystemExit("matrix needs --output FOLDER")
        print(json.dumps(asyncio.run(matrix(chosen, args.output, workdir, quick=quick)), indent=2))
    finally:
        if args.keep:
            say(f"kept {workdir}")
        elif args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
