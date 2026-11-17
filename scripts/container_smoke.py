#!/usr/bin/env python3
"""Check a running Tine container from this machine, with one command.

Without ``--send`` nothing here costs money. It checks that:

- ``GET /health`` answers, and prints which routing and shadow mode are on;
- a wrong tenant token, and no token, are refused with 401;
- ``GET /v1/savings`` answers for the tenant whose token is in ``--token-file``;
- ``GET /v1/shadow`` answers, when ``/health`` says shadow mode is sampling.

With ``--send N --model MODEL --yes-paid`` it then sends N tiny requests to
``POST /v1/messages`` (the Anthropic route), the last one streamed when N is 2 or more.
THESE ARE PAID CALLS: the container answers them with its own provider key, on that
provider account. For each it prints the status, the model that served it
(``x-tine-model``), the route (``x-tine-route``) and the request id, then reads
``/v1/savings`` again, and ``/v1/shadow`` if shadow mode is on. Sending stops at the
first request that fails.

It prints one line per check as it goes, PASS, FAIL or SKIP, and exits 1 if any check
failed. The token is read from the file and sent only to ``--base-url``; it is never
printed.

Standard library only, so the Mac's own ``python3`` runs it.
"""

from __future__ import annotations

import argparse
import http.client
import json
import secrets
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
MAX_SEND = 50
NAME_WIDTH = len("request 50 (streamed)")
# How long to wait for the cost of a streamed request, which is written when its stream ends.
SETTLE_SECONDS = 5.0


class Unreachable(Exception):
    """No HTTP reply came back."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is reported as the reply it is. Following one would send the tenant
    token to whatever host it names."""

    def redirect_request(self, *args, **kwargs):
        return None


class Reply:
    def __init__(self, status: int, headers, body: bytes) -> None:
        self.status, self.headers, self.body = status, headers, body

    def json(self):
        try:
            return json.loads(self.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def error(self) -> str:
        """The reply's own error message, shortened, for a failed check."""
        payload = self.json()
        found = payload.get("error", payload.get("detail")) if isinstance(payload, dict) else None
        if isinstance(found, dict):
            found = found.get("message")
        text = found if isinstance(found, str) else self.body[:160].decode("utf-8", "replace")
        return " ".join(text.split())[:160]


class Proxy:
    """Requests to one base URL. ``token`` is sent as a bearer token unless a call names another."""

    def __init__(self, base_url: str, token: str, timeout: float) -> None:
        self.base, self.token, self.timeout = base_url.rstrip("/"), token, timeout
        handlers = [_NoRedirect()]
        if urlsplit(self.base).hostname in ("127.0.0.1", "localhost", "::1"):
            # A proxy variable in this shell must not carry a loopback request away.
            handlers.append(urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    def call(self, method: str, path: str, *, token: str | None = "", body: dict | None = None) -> Reply:
        """``token``: "" for the tenant's own, None for no token, or another string."""
        headers = {"accept": "application/json"}
        token = self.token if token == "" else token
        if token is not None:
            headers["authorization"] = f"Bearer {token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers.update({"content-type": "application/json", "anthropic-version": "2023-06-01"})
        request = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return Reply(response.status, response.headers, response.read())
        except urllib.error.HTTPError as error:
            try:
                return Reply(error.code, error.headers, error.read())
            finally:
                error.close()
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            reason = getattr(error, "reason", None) or error
            raise Unreachable(f"{type(reason).__name__}: {reason}") from None


def usd(value) -> str:
    return "unknown" if not isinstance(value, (int, float)) or isinstance(value, bool) else f"${value:.6f}"


def read_token(path: Path) -> str:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit(f"container_smoke: --token-file {path} could not be read ({type(error).__name__}). "
                         "scripts/container_env.py writes deploy/client-token.txt.") from None
    if len(lines) != 1 or any(character.isspace() for character in lines[0]):
        raise SystemExit(f"container_smoke: --token-file {path} must hold the tenant token on one line and nothing else")
    return lines[0]


def wait_for_health(proxy: Proxy, seconds: float) -> Reply:
    """``GET /health``, tried again until it answers or ``seconds`` have passed: a
    container needs a few seconds after ``docker compose up -d`` before it listens."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return proxy.call("GET", "/health", token=None)
        except Unreachable:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)


def savings_detail(report: dict) -> str:
    return (f"tenant {report.get('tenant')!r}, {report.get('requests')} request(s), known cost "
            f"{usd(report.get('known_cost_total_usd'))}, answer cost unknown for "
            f"{report.get('unknown_answer_cost_requests')}")


def shadow_detail(report: dict) -> str:
    return (f"sampling: {str(report.get('sampling'))[:60]}; {report.get('shadowed_requests')} shadowed, "
            f"{report.get('policy_would_accept')} would be accepted, shadow spend {usd(report.get('shadow_spend_usd'))}")


def run(args, out) -> list[tuple[str, str, str]]:
    """Every check in order, as (name, result, detail) rows."""
    token = read_token(args.token_file)
    proxy = Proxy(args.base_url, token, args.timeout)
    rows: list[tuple[str, str, str]] = []

    def add(name: str, result: str, detail: str) -> None:
        # Printed as it happens: a paid request can take a while. The token never is.
        rows.append((name, result, detail.replace(token, "<token>")))
        out(f"{name:<{NAME_WIDTH}}  {result}  {rows[-1][2]}")

    out(f"Checking the Tine proxy at {proxy.base}. Token read from {args.token_file} (not shown).")
    host = urlsplit(proxy.base)
    if host.scheme != "https" and host.hostname not in ("127.0.0.1", "localhost", "::1"):
        out("Warning: this is plain HTTP to another machine. The token crosses the network unencrypted.")
    out("")

    try:
        health = wait_for_health(proxy, args.wait)
    except Unreachable as error:
        add("health", FAIL, f"no reply from {proxy.base}/health after {args.wait:g} s ({error}). Is the container "
                            "up? Try: docker compose ps, then docker compose logs tine")
        for name in ("wrong token refused", "savings"):
            add(name, SKIP, "the proxy did not answer")
        return rows
    state = health.json()
    if health.status != 200 or not isinstance(state, dict) or state.get("status") != "ok":
        add("health", FAIL, f"status {health.status}, not the proxy's health reply: {health.error()}")
        state = {}
    else:
        add("health", PASS, f"routing {state.get('routing')}, shadow {state.get('shadow')}, "
                            f"storage {state.get('storage')}")

    try:
        wrong = proxy.call("GET", "/v1/savings", token="wrong-" + secrets.token_urlsafe(16))
        absent = proxy.call("GET", "/v1/savings", token=None)
        add("wrong token refused", PASS if wrong.status == absent.status == 401 else FAIL,
            f"{wrong.status} for a wrong token, {absent.status} for no token" +
            ("" if wrong.status == absent.status == 401 else "; expected 401 for both"))

        before = proxy.call("GET", "/v1/savings")
        report = before.json()
        known = before.status == 200 and isinstance(report, dict) and isinstance(report.get("requests"), int)
        if known:
            add("savings", PASS, savings_detail(report))
        elif before.status == 401:
            add("savings", FAIL, "401: the proxy does not know the token in the token file. If deploy/tine.env was "
                                 "rewritten after the container started, run: docker compose up -d --force-recreate")
        else:
            add("savings", FAIL, f"status {before.status}: {before.error()}")

        shadowing = state.get("shadow") == "sampling"
        if shadowing:
            shadow = proxy.call("GET", "/v1/shadow")
            add("shadow report", PASS if shadow.status == 200 and isinstance(shadow.json(), dict) else FAIL,
                shadow_detail(shadow.json()) if shadow.status == 200 and isinstance(shadow.json(), dict)
                else f"status {shadow.status}: {shadow.error()}")
        else:
            add("shadow report", SKIP, "shadow mode is off, so there is no /v1/shadow to read")

        if args.send and not known:
            add("paid requests", SKIP, "not sent: the savings check failed, so the token does not work")
        elif args.send:
            send(args, proxy, report, shadowing, add, out)
    except Unreachable as error:
        add("connection", FAIL, f"the proxy stopped answering: {error}")
    return rows


def count(report: dict, name: str) -> int:
    value = report.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def send(args, proxy: Proxy, before: dict, shadowing: bool, add, out) -> None:
    out("")
    out(f"PAID CALLS: sending {args.send} request(s) to {proxy.base}/v1/messages. The container answers them "
        "with its own provider key. They cost real money on that provider account.")
    out("")
    nonce = secrets.token_hex(4)
    recorded, all_ok = 0, True
    for number in range(1, args.send + 1):
        streamed = args.send >= 2 and number == args.send
        name = f"request {number}" + (" (streamed)" if streamed else "")
        if not all_ok:
            add(name, SKIP, "not sent: an earlier request failed")
            continue
        body = {"model": args.model, "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": f"Smoke test {nonce}-{number}. Reply with the one word: ok"}]}
        if streamed:
            body["stream"] = True
        reply = proxy.call("POST", "/v1/messages", body=body)
        headers = reply.headers
        detail = (f"status {reply.status}, served by {headers.get('x-tine-model', '-')}, "
                  f"route {headers.get('x-tine-route', '-')}, id {headers.get('x-tine-request-id', '-')}, "
                  f"ledger {headers.get('x-tine-ledger', '-')}")
        recorded += headers.get("x-tine-ledger") == "recorded"
        problems = []
        if reply.status != 200:
            problems.append(reply.error())
        elif streamed and (not headers.get("content-type", "").startswith("text/event-stream")
                           or b"data:" not in reply.body):
            problems.append("the reply is not an event stream")
        elif not streamed and not isinstance(reply.json(), dict):
            problems.append("the reply is not a JSON object")
        if reply.status == 200 and headers.get("x-tine-ledger") != "recorded":
            problems.append("the answer was not written to the ledger: is the /data volume writable by uid 10001?")
        if "x-tine-request-id" not in headers:
            problems.append("no x-tine-request-id header: did the Tine proxy answer this?")
        all_ok = not problems
        add(name, PASS if all_ok else FAIL, detail + ("" if all_ok else ". " + "; ".join(problems)))

    # A streamed request's cost is written when its stream ends, a moment after the reply:
    # read the report again for a short while before calling a cost missing.
    deadline = time.monotonic() + (SETTLE_SECONDS if all_ok else 0)
    while True:
        after = proxy.call("GET", "/v1/savings")
        report = after.json() if after.status == 200 and isinstance(after.json(), dict) else {}
        grew = count(report, "requests") - before["requests"]
        unknown = count(report, "unknown_answer_cost_requests") - count(before, "unknown_answer_cost_requests")
        if not report or (grew >= recorded and unknown <= 0) or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    if not report:
        add("savings after", FAIL, f"status {after.status}: {after.error()}")
        return
    add("savings after", PASS if grew >= recorded else FAIL,
        f"requests {before['requests']} -> {report.get('requests')}, known cost "
        f"{usd(before.get('known_cost_total_usd'))} -> {usd(report.get('known_cost_total_usd'))}"
        + ("" if grew >= recorded else f"; {recorded} request(s) were marked recorded"))
    if not all_ok:
        add("costs recorded", SKIP, "a request failed")
    elif unknown > 0:
        add("costs recorded", FAIL, f"{unknown} of the request(s) sent have no answer cost. Give the model that "
                                    "served them an input and an output price in TINE_MODEL_PRICES")
    else:
        add("costs recorded", PASS, "every request sent has an answer cost")
    if shadowing:
        shadow = proxy.call("GET", "/v1/shadow")
        fine = shadow.status == 200 and isinstance(shadow.json(), dict)
        add("shadow report after", PASS if fine else FAIL,
            shadow_detail(shadow.json()) + ". Shadow runs finish after the reply: read it again in a minute"
            if fine else f"status {shadow.status}: {shadow.error()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="where the proxy listens")
    parser.add_argument("--token-file", type=Path, default=ROOT / "deploy" / "client-token.txt",
                        help="file holding the tenant token; default deploy/client-token.txt")
    parser.add_argument("--send", type=int, default=0, metavar="N",
                        help="PAID: send N tiny requests through the proxy; needs --model and --yes-paid")
    parser.add_argument("--model", help="model name to put in the requests; the proxy decides which model answers")
    parser.add_argument("--yes-paid", action="store_true",
                        help="confirm that --send makes real calls on the container's provider account")
    parser.add_argument("--max-tokens", type=int, default=16, help="max_tokens of each request sent; default 16")
    parser.add_argument("--wait", type=float, default=30.0,
                        help="seconds to keep trying /health while the container starts; default 30")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for one reply; default 120")
    args = parser.parse_args(argv)
    if urlsplit(args.base_url).scheme not in ("http", "https") or not urlsplit(args.base_url).hostname:
        parser.error("--base-url must be an http(s) URL")
    if args.send < 0 or args.send > MAX_SEND:
        parser.error(f"--send must be from 0 to {MAX_SEND}: this is a smoke test")
    if args.send and not args.model:
        parser.error("--send needs --model MODEL")
    if args.send and not args.yes_paid:
        parser.error("--send makes PAID calls on the provider account whose key is in the container. "
                     "Nothing was sent. Add --yes-paid to go ahead")
    if args.max_tokens < 1 or args.wait < 0 or args.timeout <= 0:
        parser.error("--max-tokens must be at least 1, --wait 0 or more and --timeout above 0")

    def out(text: str) -> None:
        print(text, flush=True)

    rows = run(args, out)
    failed = [name for name, result, _ in rows if result == FAIL]
    out("")
    out(f"{len(failed)} check(s) failed: {', '.join(failed)}." if failed else "All checks passed." + (
        "" if args.send else " No model was called: add --send N --model MODEL --yes-paid for paid requests."))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
