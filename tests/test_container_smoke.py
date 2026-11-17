"""scripts/container_smoke.py against the real proxy on a local port.

The proxy app is served by uvicorn on 127.0.0.1, as it is in the container. Its
provider is a stand-in on a second local port that answers like the Anthropic
Messages API, streamed or not, with made-up token counts. No request leaves this
machine and no model is called.
"""

import importlib.util
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import uvicorn

from tine.ledger import Ledger
from tine.proxy import ProxyConfig, create_app

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("container_smoke", ROOT / "scripts" / "container_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)

TOKEN = "tenant-token-" + "k3" * 16
FRONTIER, CHEAP = "frontier-model", "cheap-model"
PRICES = {f"anthropic:{FRONTIER}": {"input_per_million": 5.0, "output_per_million": 25.0},
          f"anthropic:{CHEAP}": {"input_per_million": 1.0, "output_per_million": 5.0}}
POLICY = {"version": 1, "tenant": "smoke", "task_type": "quiz", "frontier": FRONTIER,
          "stages": [{"models": [CHEAP], "check": "none"}],
          "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03,
                        "statement": "made up for a test"}}
STREAM_EVENTS = [
    ("message_start", {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 12,
                                                                                    "output_tokens": 1}}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
    ("message_stop", {"type": "message_stop"}),
]


class StandInProvider:
    """Answers POST /v1/messages like the Anthropic API and remembers what it was sent."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.keys: set[str] = set()
        self.status = 200
        provider = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                provider.requests.append(body)
                provider.keys.add(self.headers.get("x-api-key", ""))
                if provider.status != 200:
                    kind, payload = "application/json", json.dumps({"type": "error", "error": {
                        "type": "authentication_error", "message": "stand-in says no"}}).encode()
                elif body.get("stream"):
                    kind = "text/event-stream"
                    payload = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n"
                                      for name, data in STREAM_EVENTS).encode()
                else:
                    kind, payload = "application/json", json.dumps({
                        "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                        "usage": {"input_tokens": 12, "output_tokens": 3}}).encode()
                self.send_response(provider.status)
                self.send_header("content-type", kind)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def provider():
    stand_in = StandInProvider()
    yield stand_in
    stand_in.close()


@pytest.fixture
def serve(provider, tmp_path, monkeypatch):
    """Start the real proxy on a free local port; returns its base URL."""
    # The proxy's own HTTP client must reach the stand-in directly, whatever this shell sets.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    servers = []

    def start(ledger=None, **changes) -> str:
        values = dict(tenant_keys={TOKEN: "smoke"}, anthropic_api_key="provider-key-for-the-stand-in",
                      anthropic_frontier_model=FRONTIER, anthropic_base_url=provider.url, prices=PRICES,
                      ledger_path=str(tmp_path / f"ledger-{len(servers)}.sqlite3"), timeout_seconds=10.0)
        values.update(changes)
        server = uvicorn.Server(uvicorn.Config(create_app(ProxyConfig(**values), ledger=ledger), host="127.0.0.1",
                                               port=0, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 20
        while not server.started:
            assert time.monotonic() < deadline and thread.is_alive(), "the proxy did not start"
            time.sleep(0.01)
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "client-token.txt"
    path.write_text(TOKEN + "\n", encoding="utf-8")
    return path


def run(capsys, base_url, token_file, *extra) -> tuple[int, str]:
    code = smoke.main(["--base-url", base_url, "--token-file", str(token_file), "--wait", "0", "--timeout", "20",
                       *extra])
    printed = "".join(capsys.readouterr())
    assert TOKEN not in printed, "the token must never be printed"
    return code, printed


def results(printed: str) -> dict[str, str]:
    """Check name -> PASS, FAIL or SKIP, from the table at the end."""
    found = {}
    for line in printed.splitlines():
        for result in (smoke.PASS, smoke.FAIL, smoke.SKIP):
            name, separator, _ = line.partition(f"  {result}  ")
            if separator:
                found[name.strip()] = result
    return found


def test_free_checks_pass_and_call_no_model(serve, provider, token_file, capsys):
    code, printed = run(capsys, serve(), token_file)
    assert code == 0, printed
    assert results(printed) == {"health": "PASS", "wrong token refused": "PASS", "savings": "PASS",
                                "shadow report": "SKIP"}
    assert "routing frontier-default, shadow off" in printed
    assert "401 for a wrong token, 401 for no token" in printed
    assert "tenant 'smoke', 0 request(s)" in printed
    assert "No model was called" in printed and provider.requests == []


def test_a_token_the_proxy_does_not_know_fails(serve, provider, tmp_path, capsys):
    other = tmp_path / "other-token.txt"
    other.write_text("not-the-tenant-token\n", encoding="utf-8")
    code, printed = run(capsys, serve(), other, "--send", "2", "--model", FRONTIER, "--yes-paid")
    assert code == 1
    found = results(printed)
    assert found["health"] == "PASS" and found["savings"] == "FAIL" and found["paid requests"] == "SKIP"
    assert "docker compose up -d" in printed and "not-the-tenant-token" not in printed
    assert provider.requests == [], "nothing is sent when the token does not work"


def test_nothing_listening_fails_with_a_hint(token_file, capsys):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    code, printed = run(capsys, f"http://127.0.0.1:{free_port}", token_file)
    assert code == 1
    assert results(printed) == {"health": "FAIL", "wrong token refused": "SKIP", "savings": "SKIP"}
    assert "docker compose ps" in printed and "1 check(s) failed" in printed


def test_health_is_tried_again_while_the_container_starts(serve, token_file, capsys, monkeypatch):
    base_url = serve()
    real, calls = smoke.Proxy.call, []

    def flaky(self, method, path, **kwargs):
        calls.append(path)
        if len(calls) <= 2:
            raise smoke.Unreachable("ConnectionRefusedError: not listening yet")
        return real(self, method, path, **kwargs)

    monkeypatch.setattr(smoke.Proxy, "call", flaky)
    monkeypatch.setattr(smoke.time, "sleep", lambda seconds: None)
    code = smoke.main(["--base-url", base_url, "--token-file", str(token_file), "--wait", "30"])
    assert code == 0 and calls[:3] == ["/health"] * 3
    capsys.readouterr()


@pytest.mark.parametrize("arguments,expected", [
    (["--send", "2", "--model", FRONTIER], "PAID"),
    (["--send", "2", "--yes-paid"], "--model"),
    (["--send", "500", "--model", FRONTIER, "--yes-paid"], "smoke test"),
    (["--base-url", "ftp://example.invalid"], "http(s)"),
])
def test_paid_requests_need_an_explicit_yes(provider, token_file, capsys, arguments, expected):
    with pytest.raises(SystemExit) as stopped:
        smoke.main(["--token-file", str(token_file), *arguments])
    assert stopped.value.code == 2
    assert expected in capsys.readouterr().err and provider.requests == []


def test_send_shows_each_request_and_reads_savings_again(serve, provider, token_file, capsys):
    code, printed = run(capsys, serve(), token_file, "--send", "3", "--model", "any-name", "--yes-paid")
    assert code == 0, printed
    assert "PAID CALLS: sending 3 request(s)" in printed and "cost real money" in printed
    found = results(printed)
    assert found == {"health": "PASS", "wrong token refused": "PASS", "savings": "PASS", "shadow report": "SKIP",
                     "request 1": "PASS", "request 2": "PASS", "request 3 (streamed)": "PASS",
                     "savings after": "PASS", "costs recorded": "PASS"}
    # The proxy, not the caller, chose the model; one request was streamed.
    assert [body["model"] for body in provider.requests] == [FRONTIER] * 3
    assert [bool(body.get("stream")) for body in provider.requests] == [False, False, True]
    assert len({body["messages"][0]["content"] for body in provider.requests}) == 3
    assert all(body["max_tokens"] == 16 for body in provider.requests)
    assert provider.keys == {"provider-key-for-the-stand-in"}, "the caller's token is not sent to the provider"
    lines = [line for line in printed.splitlines() if line.startswith("request ")]
    assert len(lines) == 3
    for line in lines:
        assert f"status 200, served by {FRONTIER}, route -, id " in line and "ledger recorded" in line
    assert len({line.split("id ")[1].split(",")[0] for line in lines}) == 3, "each request has its own id"
    assert "requests 0 -> 3" in printed and "known cost $0.000000 -> $0.000405" in printed


def test_one_request_is_not_streamed(serve, provider, token_file, capsys):
    code, printed = run(capsys, serve(), token_file, "--send", "1", "--model", FRONTIER, "--yes-paid",
                        "--max-tokens", "8")
    assert code == 0 and results(printed)["request 1"] == "PASS"
    assert [(body.get("stream"), body["max_tokens"]) for body in provider.requests] == [(None, 8)]


def test_a_route_policy_shows_in_health_and_in_each_request(serve, provider, token_file, capsys):
    code, printed = run(capsys, serve(route_policies={"smoke": {"anthropic": POLICY}}), token_file,
                        "--send", "2", "--model", FRONTIER, "--yes-paid")
    assert code == 0, printed
    assert "routing policy, shadow off" in printed
    assert f"served by {CHEAP}, route stage-0" in printed
    # Streamed requests are never routed by a policy: the frontier answers them.
    assert [body["model"] for body in provider.requests] == [CHEAP, FRONTIER]
    assert results(printed)["request 2 (streamed)"] == "PASS"


def test_shadow_mode_is_reported_when_it_is_sampling(serve, provider, token_file, capsys):
    base_url = serve(shadow_policies={"smoke": {"anthropic": POLICY}}, shadow_sample_rate=1.0,
                     shadow_max_cost_usd=1.0)
    code, printed = run(capsys, base_url, token_file, "--send", "2", "--model", FRONTIER, "--yes-paid")
    assert code == 0, printed
    found = results(printed)
    assert found["shadow report"] == "PASS" and found["shadow report after"] == "PASS"
    assert "shadow sampling" in printed and "sampling: on" in printed


def test_a_provider_error_fails_and_stops_the_sending(serve, provider, token_file, capsys):
    provider.status = 401
    code, printed = run(capsys, serve(), token_file, "--send", "3", "--model", FRONTIER, "--yes-paid")
    assert code == 1
    found = results(printed)
    assert found["request 1"] == "FAIL" and found["request 2"] == "SKIP" and found["request 3 (streamed)"] == "SKIP"
    assert found["costs recorded"] == "SKIP"
    assert "status 401" in printed and "stand-in says no" in printed
    assert len(provider.requests) == 1, "sending stops at the first failure"


def test_an_unpriced_model_fails_the_cost_check(serve, provider, token_file, capsys, monkeypatch):
    monkeypatch.setattr(smoke, "SETTLE_SECONDS", 0.2)
    code, printed = run(capsys, serve(prices={}), token_file, "--send", "2", "--model", FRONTIER, "--yes-paid")
    assert code == 1
    found = results(printed)
    assert found["request 1"] == "PASS" and found["savings after"] == "PASS" and found["costs recorded"] == "FAIL"
    assert "TINE_MODEL_PRICES" in printed


def test_an_answer_the_ledger_could_not_record_fails(serve, provider, token_file, tmp_path, capsys):
    """What a /data volume the container cannot write to looks like from outside."""

    class BrokenLedger(Ledger):
        def record(self, **values):
            raise OSError("read-only file system")

    code, printed = run(capsys, serve(ledger=BrokenLedger(tmp_path / "broken.sqlite3")), token_file,
                        "--send", "2", "--model", FRONTIER, "--yes-paid")
    assert code == 1
    found = results(printed)
    assert found["request 1"] == "FAIL" and found["request 2 (streamed)"] == "SKIP"
    assert "ledger unavailable" in printed and "uid 10001" in printed


def test_a_redirect_is_not_followed_with_the_token(provider, token_file, capsys):
    """A reply that points somewhere else is a failed check, not a place to send the token."""
    seen = []

    class Redirecting(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("authorization")))
            self.send_response(302)
            self.send_header("location", provider.url + "/elsewhere")
            self.send_header("content-length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirecting)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    try:
        code, printed = run(capsys, f"http://127.0.0.1:{server.server_address[1]}", token_file)
    finally:
        server.shutdown()
        server.server_close()
    assert code == 1 and results(printed)["health"] == "FAIL" and "status 302" in printed
    assert {path for path, _ in seen} <= {"/health", "/v1/savings"}, "no request went to the redirect's target"


@pytest.mark.parametrize("content", ["", "two tokens\n", "first\nsecond\n"])
def test_a_token_file_must_hold_one_token(tmp_path, capsys, content):
    path = tmp_path / "token.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        smoke.main(["--token-file", str(path)])
    assert "one line" in str(stopped.value)
    assert not content.strip() or content.strip() not in str(stopped.value)
    with pytest.raises(SystemExit, match="container_env.py"):
        smoke.main(["--token-file", str(tmp_path / "missing.txt")])
