#!/usr/bin/env python3
"""A stand-in model provider for testing the Tine container. It calls no model.

It answers ``POST /v1/messages`` in the shape of the Anthropic Messages API, with a
fixed one-word reply and made-up token counts, streamed when the request asks for a
stream. ``GET /health`` answers 200. ``scripts/container_selftest.py`` runs it as a
second container beside the proxy, so the proxy can be tested end to end without a
provider key and without a paid call.

A streamed request whose text contains ``HOLD_STREAM_OPEN`` gets a stream that never
ends: the first event, then a comment line every second. The self-test uses it to
check that the proxy still stops in time with a stream open.

Standard library only. It keeps nothing and sends nothing anywhere.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPLY_ID = "msg_selftest_standin"
HOLD_MARKER = "HOLD_STREAM_OPEN"
USAGE = {"input_tokens": 12, "output_tokens": 3}


def events(model: str) -> list:
    """The server-sent events of one streamed reply, as (name, data) pairs."""
    return [
        ("message_start", {"type": "message_start", "message": {
            "id": REPLY_ID, "type": "message", "role": "assistant", "model": model, "content": [],
            "usage": {"input_tokens": USAGE["input_tokens"], "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": USAGE["output_tokens"]}}),
        ("message_stop", {"type": "message_stop"}),
    ]


def frame(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def request_text(body: dict) -> str:
    """Every piece of text in the request's messages, joined."""
    parts = []
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(block.get("text", "") for block in content if isinstance(block, dict))
    return " ".join(part for part in parts if isinstance(part, str))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "standin": True})
        else:
            self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": "stand-in: no such path"}})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
        except ValueError:
            body = None
        if self.path != "/v1/messages" or not isinstance(body, dict):
            self._send(404 if self.path != "/v1/messages" else 400, {"type": "error", "error": {
                "type": "invalid_request_error", "message": "stand-in: it answers only POST /v1/messages with JSON"}})
            return
        model = body.get("model") if isinstance(body.get("model"), str) else "unknown"
        if not body.get("stream"):
            self._send(200, {"id": REPLY_ID, "type": "message", "role": "assistant", "model": model,
                             "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                             "usage": dict(USAGE)})
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            if HOLD_MARKER in request_text(body):
                self.wfile.write(frame(*events(model)[0]))
                while True:
                    self.wfile.write(b": still open\n\n")
                    self.wfile.flush()
                    time.sleep(1)
            for name, data in events(model):
                self.wfile.write(frame(name, data))
            self.wfile.flush()
        except OSError:
            pass  # the caller went away

    def log_message(self, *args):
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args(argv)
    server = Server((args.host, args.port), Handler)
    print(f"stand-in provider on http://{args.host}:{server.server_address[1]} (no model is called)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
