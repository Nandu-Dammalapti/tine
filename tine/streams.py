"""Server-sent event streams of the OpenAI and Anthropic chat APIs, read and written.

Three parts, none of which changes a relayed byte:

- ``SSEEvents`` splits relayed bytes into the data of each event, however the bytes
  were cut into chunks and whatever line endings they use.
- ``StreamAssembler`` builds from a stream the reply the same request would have had
  without ``stream``: what shadow mode compares with the cheaper models' answers.
- ``replay`` writes a reply that was not streamed as the events a streamed request
  expects, so that a route policy can serve a cheaper model's buffered answer to a
  caller that asked for a stream.

Only what the providers document for chat replies is read or written: text, tool
calls and, for Anthropic, thinking blocks. A reply holding anything else cannot be
replayed (``replay`` returns None) and a stream holding anything else cannot be
assembled (``StreamAssembler.payload`` returns None): such a request is left to the
frontier as it is.
"""

from __future__ import annotations

import json
import time
from typing import Any

# A longer line is skipped, not buffered.
MAX_LINE = 1 << 20
# What an assembled reply may hold, in bytes of text and tool arguments. A longer stream
# is still relayed whole; it is only not assembled.
MAX_ASSEMBLED = 8 << 20


class SSEEvents:
    """The data of each server-sent event, from bytes cut anywhere.

    ``feed`` and ``close`` return the data of the events that ended, one ``bytes`` per
    event, its ``data:`` lines joined by newlines. Comments, ``event:`` and other fields
    are dropped. A line over ``MAX_LINE`` bytes is skipped. Nothing here raises on odd
    input: a relayed stream must never be interrupted by reading it.
    """

    def __init__(self) -> None:
        self._line = bytearray()
        self._skipping = False
        self._after_cr = False
        self._data: list[bytes] = []

    def feed(self, chunk: bytes) -> list[bytes]:
        events: list[bytes] = []
        try:
            for piece in chunk.splitlines(keepends=True):
                if self._after_cr and piece == b"\n" and not self._line:
                    # The second half of a CRLF split across two chunks.
                    self._after_cr = False
                    continue
                ended = piece.endswith((b"\n", b"\r"))
                self._after_cr = piece.endswith(b"\r")
                if not self._skipping:
                    self._line += piece
                    if len(self._line) > MAX_LINE:
                        self._line.clear()
                        self._skipping = True
                if ended:
                    if not self._skipping:
                        self._take(bytes(self._line).rstrip(b"\r\n"), events)
                    self._line.clear()
                    self._skipping = False
        except Exception:
            self._line.clear()
        return events

    def close(self) -> list[bytes]:
        """The stream ended: the last event, if no blank line followed it."""
        events: list[bytes] = []
        if self._line and not self._skipping:
            self._take(bytes(self._line).rstrip(b"\r\n"), events)
        self._line.clear()
        self._take(b"", events)
        return events

    def _take(self, line: bytes, events: list[bytes]) -> None:
        if line.startswith(b"data:"):
            self._data.append(line[5:].removeprefix(b" "))
            return
        if line or not self._data:
            return
        events.append(b"\n".join(self._data))
        self._data = []


def _decode(data: bytes) -> dict | None:
    try:
        event = json.loads(data)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    return event if isinstance(event, dict) else None


class StreamAssembler:
    """The reply a streamed request would have had without ``stream``, from its events.

    Feed it the relayed bytes; ``payload()`` gives the reply once the stream has ended,
    or None when the stream was cut off, said it failed, held something this module
    does not know, or grew past ``MAX_ASSEMBLED``.
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self._events = SSEEvents()
        self._size = 0
        self._broken = False
        self._done = False
        self._closed = False
        # Anthropic
        self._message: dict[str, Any] | None = None
        self._blocks: dict[int, dict[str, Any]] = {}
        self._json: dict[int, list[str]] = {}
        # OpenAI
        self._head: dict[str, Any] = {}
        self._choices: dict[int, dict[str, Any]] = {}
        self._usage: dict[str, Any] | None = None

    def feed(self, chunk: bytes) -> None:
        if self._broken:
            return
        try:
            for data in self._events.feed(chunk):
                self._event(data)
        except Exception:
            self._broken = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for data in self._events.close():
                self._event(data)
        except Exception:
            self._broken = True

    def _grow(self, text: Any) -> str:
        if not isinstance(text, str):
            raise ValueError("not text")
        self._size += len(text)
        if self._size > MAX_ASSEMBLED:
            raise ValueError("too long to assemble")
        return text

    def _event(self, data: bytes) -> None:
        if self._broken or self._done and self.provider == "openai":
            return
        if self.provider == "openai":
            if data.strip() == b"[DONE]":
                self._done = True
                return
            event = _decode(data)
            if event is None:
                raise ValueError("not an event")
            if "error" in event:
                raise ValueError("the stream reported an error")
            self._openai(event)
            return
        event = _decode(data)
        if event is None:
            raise ValueError("not an event")
        self._anthropic(event)

    def _openai(self, event: dict) -> None:
        for key in ("id", "model", "created", "system_fingerprint", "service_tier"):
            if key in event and key not in self._head:
                self._head[key] = event[key]
        if isinstance(event.get("usage"), dict):
            self._usage = event["usage"]
        for choice in event.get("choices") or []:
            index = choice.get("index", 0)
            if not isinstance(index, int):
                raise ValueError("bad choice index")
            built = self._choices.setdefault(index, {"role": None, "content": None, "tool_calls": {},
                                                     "refusal": None, "finish_reason": None})
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise ValueError("bad delta")
            unknown = set(delta) - {"role", "content", "tool_calls", "refusal", "function_call"}
            if unknown:
                raise ValueError(f"unknown delta fields {sorted(unknown)}")
            if delta.get("function_call"):
                raise ValueError("legacy function calls are not assembled")
            if delta.get("role"):
                built["role"] = delta["role"]
            for name in ("content", "refusal"):
                if delta.get(name) is not None:
                    built[name] = (built[name] or "") + self._grow(delta[name])
            for call in delta.get("tool_calls") or []:
                slot = call.get("index", 0)
                if not isinstance(slot, int):
                    raise ValueError("bad tool call index")
                tool = built["tool_calls"].setdefault(slot, {"id": None, "type": "function",
                                                             "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    tool["id"] = call["id"]
                if call.get("type"):
                    tool["type"] = call["type"]
                function = call.get("function") or {}
                if function.get("name"):
                    tool["function"]["name"] += self._grow(function["name"])
                if function.get("arguments"):
                    tool["function"]["arguments"] += self._grow(function["arguments"])
            if choice.get("finish_reason") is not None:
                built["finish_reason"] = choice["finish_reason"]

    def _anthropic(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message")
            if not isinstance(message, dict):
                raise ValueError("bad message_start")
            self._message = {**message, "content": []}
        elif kind == "content_block_start":
            index, block = event.get("index"), event.get("content_block")
            if not isinstance(index, int) or not isinstance(block, dict):
                raise ValueError("bad content_block_start")
            if block.get("type") not in ("text", "tool_use", "thinking", "redacted_thinking"):
                raise ValueError(f"block type {block.get('type')!r} is not assembled")
            self._blocks[index] = dict(block)
        elif kind == "content_block_delta":
            index, delta = event.get("index"), event.get("delta")
            block = self._blocks.get(index)
            if block is None or not isinstance(delta, dict):
                raise ValueError("a delta for no block")
            which = delta.get("type")
            if which == "text_delta":
                block["text"] = block.get("text", "") + self._grow(delta.get("text"))
            elif which == "input_json_delta":
                self._json.setdefault(index, []).append(self._grow(delta.get("partial_json")))
            elif which == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + self._grow(delta.get("thinking"))
            elif which == "signature_delta":
                block["signature"] = block.get("signature", "") + self._grow(delta.get("signature"))
            else:
                raise ValueError(f"delta type {which!r} is not assembled")
        elif kind == "content_block_stop":
            index = event.get("index")
            if index in self._json:
                text = "".join(self._json.pop(index))
                self._blocks[index]["input"] = json.loads(text) if text else {}
        elif kind == "message_delta":
            if self._message is None:
                raise ValueError("message_delta before message_start")
            delta = event.get("delta") or {}
            for key in ("stop_reason", "stop_sequence"):
                if key in delta:
                    self._message[key] = delta[key]
            if isinstance(event.get("usage"), dict):
                self._message["usage"] = {**(self._message.get("usage") or {}), **event["usage"]}
        elif kind == "message_stop":
            self._done = True
        elif kind == "error":
            raise ValueError("the stream reported an error")
        elif kind != "ping":
            raise ValueError(f"event type {kind!r} is not assembled")

    def payload(self) -> dict | None:
        """The assembled reply, or None (see the class)."""
        self.close()
        if self._broken or not self._done:
            return None
        if self.provider == "anthropic":
            if self._message is None or self._json:
                return None
            return {**self._message, "content": [self._blocks[index] for index in sorted(self._blocks)]}
        if not self._choices:
            return None
        choices = []
        for index in sorted(self._choices):
            built = self._choices[index]
            message: dict[str, Any] = {"role": built["role"] or "assistant", "content": built["content"]}
            if built["refusal"] is not None:
                message["refusal"] = built["refusal"]
            if built["tool_calls"]:
                message["tool_calls"] = [built["tool_calls"][slot] for slot in sorted(built["tool_calls"])]
            choices.append({"index": index, "message": message, "finish_reason": built["finish_reason"]})
        return {**self._head, "object": "chat.completion", "choices": choices,
                **({"usage": self._usage} if self._usage is not None else {})}


def _sse(data: dict, event: str | None = None) -> bytes:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (f"event: {event}\n" if event else "").encode("utf-8") + f"data: {text}\n\n".encode("utf-8")


def _replay_anthropic(payload: dict) -> bytes | None:
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    start_usage = {key: value for key, value in usage.items() if key != "output_tokens"}
    message = {key: value for key, value in payload.items() if key not in ("content", "stop_reason", "stop_sequence",
                                                                           "usage")}
    message.setdefault("type", "message")
    message.setdefault("role", "assistant")
    parts = [_sse({"type": "message_start", "message": {**message, "content": [], "stop_reason": None,
                                                         "stop_sequence": None,
                                                         "usage": {**start_usage, "output_tokens": 0}}},
                  "message_start")]
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            return None
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            opening, deltas = {**block, "text": ""}, [{"type": "text_delta", "text": block["text"]}]
        elif kind == "tool_use" and isinstance(block.get("input"), dict):
            opening = {**block, "input": {}}
            deltas = [{"type": "input_json_delta",
                       "partial_json": json.dumps(block["input"], ensure_ascii=False, allow_nan=False)}]
        elif kind == "thinking" and isinstance(block.get("thinking"), str):
            opening = {**block, "thinking": "", "signature": ""}
            deltas = [{"type": "thinking_delta", "thinking": block["thinking"]}]
            if isinstance(block.get("signature"), str) and block["signature"]:
                deltas.append({"type": "signature_delta", "signature": block["signature"]})
        elif kind == "redacted_thinking":
            opening, deltas = dict(block), []
        else:
            return None
        parts.append(_sse({"type": "content_block_start", "index": index, "content_block": opening},
                          "content_block_start"))
        for delta in deltas:
            parts.append(_sse({"type": "content_block_delta", "index": index, "delta": delta}, "content_block_delta"))
        parts.append(_sse({"type": "content_block_stop", "index": index}, "content_block_stop"))
    final_usage = {"output_tokens": usage["output_tokens"]} if "output_tokens" in usage else {}
    parts.append(_sse({"type": "message_delta",
                       "delta": {"stop_reason": payload.get("stop_reason"),
                                 "stop_sequence": payload.get("stop_sequence")},
                       "usage": final_usage}, "message_delta"))
    parts.append(_sse({"type": "message_stop"}, "message_stop"))
    return b"".join(parts)


def _replay_openai(payload: dict, include_usage: bool) -> bytes | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    head = {key: payload[key] for key in ("id", "model", "system_fingerprint", "service_tier") if key in payload}
    head.update(object="chat.completion.chunk", created=payload.get("created", int(time.time())))
    extra = {"usage": None} if include_usage else {}
    parts = []
    for position, choice in enumerate(choices):
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            return None
        message = choice["message"]
        if set(message) - {"role", "content", "tool_calls", "refusal", "annotations"} or message.get("annotations"):
            return None
        index = choice.get("index", position)
        opening: dict[str, Any] = {"role": message.get("role") or "assistant"}
        if message.get("refusal") is None:
            opening["content"] = ""
        parts.append(_sse({**head, "choices": [{"index": index, "delta": opening, "finish_reason": None}], **extra}))
        if isinstance(message.get("content"), str) and message["content"]:
            parts.append(_sse({**head, "choices": [{"index": index, "delta": {"content": message["content"]},
                                                    "finish_reason": None}], **extra}))
        if isinstance(message.get("refusal"), str):
            parts.append(_sse({**head, "choices": [{"index": index, "delta": {"refusal": message["refusal"]},
                                                    "finish_reason": None}], **extra}))
        for slot, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                return None
            delta = {"tool_calls": [{"index": slot, "id": call.get("id"), "type": call.get("type", "function"),
                                     "function": {"name": function.get("name", ""),
                                                  "arguments": function.get("arguments", "")}}]}
            parts.append(_sse({**head, "choices": [{"index": index, "delta": delta, "finish_reason": None}], **extra}))
        parts.append(_sse({**head, "choices": [{"index": index, "delta": {},
                                                "finish_reason": choice.get("finish_reason")}], **extra}))
    if include_usage:
        parts.append(_sse({**head, "choices": [], "usage": payload.get("usage")}))
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


def replay(provider: str, payload: Any, *, include_usage: bool = False) -> bytes | None:
    """A reply that was not streamed, as the events of a streamed one, or None when it
    holds something this module does not write. ``include_usage`` is OpenAI's
    ``stream_options.include_usage``: whether a last chunk carries the token counts.
    Anthropic streams always carry them."""
    if not isinstance(payload, dict):
        return None
    try:
        return _replay_openai(payload, include_usage) if provider == "openai" else _replay_anthropic(payload)
    except (TypeError, ValueError):
        return None


def unstreamed(body: dict) -> dict:
    """The same request asked for without a stream: what a cheaper stage is sent."""
    return {key: value for key, value in body.items() if key not in ("stream", "stream_options")}


def wants_usage(body: dict) -> bool:
    """Whether an OpenAI caller asked for the token counts at the end of its stream."""
    options = body.get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") is True
