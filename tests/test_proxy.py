import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from tine.ledger import Ledger
from tine.proxy import ProxyConfig, _cost, _StreamUsage, _usage, create_app


def config(**changes):
    values = dict(
        tenant_keys={"tenant-token-a": "a", "tenant-token-b": "b"},
        openai_api_key="provider-secret-o",
        anthropic_api_key="provider-secret-a",
        openai_frontier_model="frontier",
        anthropic_frontier_model="claude-frontier",
    )
    values.update(changes)
    return ProxyConfig(**values)


def run(coroutine):
    return asyncio.run(coroutine)


@pytest.mark.parametrize("changes,field", [
    ({"tenant_keys": ["token"]}, "TINE_TENANT_KEYS"),
    ({"allowed_models": "cheap"}, "TINE_ALLOWED_MODELS"),
    ({"allowed_models": {"openai": "cheap-v2"}}, "TINE_ALLOWED_MODELS"),
    ({"allowed_models": {"openai": {"cheap": True}}}, "TINE_ALLOWED_MODELS"),
    ({"allowed_models": {"openai": [None]}}, "TINE_ALLOWED_MODELS"),
    ({"allowed_models": {"openai": [""]}}, "TINE_ALLOWED_MODELS"),
    ({"allowed_models": {"unknown": ["cheap"]}}, "TINE_ALLOWED_MODELS"),
    ({"prices": []}, "prices"),
    ({"prices": {"openai:frontier": []}}, "prices"),
    ({"prices": {"openai:frontier": {"input_per_million": 10**400}}}, "prices"),
    ({"timeout_seconds": True}, "timeout_seconds"),
    ({"timeout_seconds": "30"}, "timeout_seconds"),
    ({"timeout_seconds": 10**400}, "timeout_seconds"),
    ({"verified_cache": "false"}, "verified_cache"),
    ({"openai_base_url": "https://example.com:invalid"}, "base URLs"),
    ({"openai_base_url": "https://example.com:99999"}, "base URLs"),
    ({"openai_base_url": None}, "base URLs"),
    ({"max_body_bytes": 0}, "TINE_MAX_BODY_BYTES"),
    ({"max_body_bytes": True}, "TINE_MAX_BODY_BYTES"),
    ({"max_body_bytes": 1.5}, "TINE_MAX_BODY_BYTES"),
    ({"denied_fields": ["store"]}, "TINE_DENIED_FIELDS"),
    ({"denied_fields": {"openai": "store"}}, "TINE_DENIED_FIELDS"),
    ({"denied_fields": {"openai": [""]}}, "TINE_DENIED_FIELDS"),
    ({"denied_fields": {"gemini": ["store"]}}, "TINE_DENIED_FIELDS"),
    ({"anthropic_beta_allowlist": "a,b"}, "TINE_ANTHROPIC_BETA_ALLOWLIST"),
    ({"anthropic_beta_allowlist": [""]}, "TINE_ANTHROPIC_BETA_ALLOWLIST"),
    ({"anthropic_beta_allowlist": ["a,b"]}, "TINE_ANTHROPIC_BETA_ALLOWLIST"),
])
def test_invalid_configuration_fails_before_opening_ledger(changes, field, monkeypatch):
    def unexpected_ledger(*args, **kwargs):
        pytest.fail("Invalid configuration opened a ledger")

    monkeypatch.setattr("tine.proxy.Ledger", unexpected_ledger)
    with pytest.raises(ValueError, match=field):
        create_app(config(**changes))


def test_valid_allowlist_requires_an_exact_model_name():
    seen = []

    def provider(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"choices": []})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(allowed_models={"openai": ["cheap-v2"]}), ledger=Ledger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                for name in ("cheap", "cheap-v2"):
                    response = await client.post(
                        "/v1/chat/completions", json={"model": name, "messages": []},
                        headers={"authorization": "Bearer tenant-token-a"},
                    )
                    assert response.status_code == 200
        assert seen == ["frontier", "cheap-v2"]

    run(scenario())


@pytest.mark.parametrize("rate,tokens", [(1e308, 3), (10**308, 3), (1e-320, 10**18)])
def test_representable_usage_cost_survives_intermediate_overflow_and_underflow(rate, tokens):
    cfg = config(prices={"openai:frontier": {"input_per_million": rate, "output_per_million": rate}})
    cfg.validate()
    result = _cost(cfg, "openai", "frontier", {
        "input_tokens": tokens, "output_tokens": tokens, "cached_tokens": 0, "cache_write_tokens": 0,
    })
    expected = float(Decimal.from_float(float(rate)) * Decimal(tokens) * 2 / Decimal(1_000_000))
    assert result == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("raw", [
    b'{"answer":"\\ud800"}', b'{"answer":NaN}', b'{"answer":1e999}',
])
def test_invalid_upstream_json_values_preserve_response_and_record_without_caching(raw):
    calls = []

    def provider(request):
        calls.append(request)
        return httpx.Response(200, content=raw, headers={"content-type": "application/json"})

    async def scenario():
        ledger = Ledger()
        if b"ud800" in raw:
            ledger.record(
                tenant="a", provider="openai", model="frontier",
                request_body={"messages": [], "model": "frontier"},
                response_body=json.loads(raw), success=True, upstream_status=200,
                cache_context={"url": "https://api.openai.com/v1/chat/completions", "headers": {}},
            )
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(), ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                for _ in range(2):
                    response = await client.post(
                        "/v1/chat/completions", json={"messages": []},
                        headers={"authorization": "Bearer tenant-token-a"},
                    )
                    assert response.status_code == 200
                    assert response.content == raw
                    assert response.headers["x-tine-ledger"] == "recorded"
                    assert "x-tine-cache" not in response.headers
                    record = ledger.get("a", response.headers["x-tine-request-id"])
                    assert record["response_body"] is None
                    assert record["cost_usd"] is None
                    ledger.update_outcome("a", record["request_id"], True)
        assert len(calls) == 2

    run(scenario())


def test_auth_frontier_default_credentials_and_json_passthrough():
    seen = []

    def provider(request):
        seen.append(request)
        assert request.headers["authorization"] == "Bearer provider-secret-o"
        payload = json.loads(request.content)
        assert payload["model"] == "frontier"
        assert payload["temperature"] == 0.1
        assert payload["messages"][0]["content"] == "hello"
        return httpx.Response(200, json={"id": "upstream-id", "choices": [{"message": {"content": "answer"}}]})

    async def scenario():
        ledger = Ledger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(), ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                body = {"model": "unapproved-cheap", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.1}
                assert (await client.post("/v1/chat/completions", json=body)).status_code == 401
                response = await client.post("/v1/chat/completions", json=body, headers={"authorization": "Bearer tenant-token-a", "x-tenant-id": "b"})
                assert response.status_code == 200
                assert response.json()["id"] == "upstream-id"
                assert response.headers["x-tine-model"] == "frontier"
                record = ledger.get("a", response.headers["x-tine-request-id"])
                assert record["requested_model"] == "unapproved-cheap"
                assert record["success"] is None
                assert record["cost_usd"] is None
                serialized = json.dumps(record)
                assert "provider-secret" not in serialized
                assert "tenant-token" not in serialized
                assert ledger.report("b")["requests"] == 0
        assert len(seen) == 1

    run(scenario())


def test_fallback_on_candidate_error_and_accounting_remains_unknown():
    attempted = []

    def provider(request):
        model = json.loads(request.content)["model"]
        attempted.append(model)
        if model == "cheap":
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    async def scenario():
        ledger = Ledger()
        cfg = config(allowed_models={"openai": ["cheap"]}, prices={"openai:frontier": {"input_per_million": 10, "output_per_million": 20}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(cfg, ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post("/v1/chat/completions", json={"model": "cheap", "messages": []}, headers={"authorization": "Bearer tenant-token-a"})
                assert response.status_code == 200
                record = ledger.get("a", response.headers["x-tine-request-id"])
                assert record["fallback_used"] == 1
                assert record["model"] == "frontier"
                assert record["cost_usd"] is None  # First attempt's bill is unknown.
        assert attempted == ["cheap", "frontier"]

    run(scenario())


def test_frontier_error_is_not_retried_and_error_details_are_not_leaked():
    attempted = []

    def provider(request):
        attempted.append(request)
        raise httpx.ReadTimeout("contains provider-secret-o", request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(), ledger=Ledger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post("/v1/chat/completions", json={"messages": []}, headers={"authorization": "Bearer tenant-token-a"})
                assert response.status_code == 502
                assert "provider-secret" not in response.text
        assert len(attempted) == 1

    run(scenario())


class SlowProvider(httpx.AsyncBaseTransport):
    """A provider whose reply takes ``seconds`` to arrive and whose address takes
    ``connect_seconds`` to reach. Nothing sleeps: it answers a client that would have
    waited that long and times out any other, as httpx itself would."""

    def __init__(self, seconds, connect_seconds=0.1):
        self.seconds, self.connect_seconds, self.timeouts = seconds, connect_seconds, []

    async def handle_async_request(self, request):
        timeout = request.extensions["timeout"]
        self.timeouts.append(timeout)
        if timeout["connect"] is not None and timeout["connect"] < self.connect_seconds:
            raise httpx.ConnectTimeout("unreachable", request=request)
        if timeout["read"] is not None and timeout["read"] < self.seconds:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "a long answer"}}]})


def ask_own_client(monkeypatch, provider, app_config):
    """One request through a proxy that opens its own provider clients, as ``tine
    serve`` does, with ``provider`` in place of the network."""
    class Wired(httpx.AsyncClient):
        def __init__(self, **options):
            super().__init__(transport=provider, **options)

    async def scenario():
        with monkeypatch.context() as patch:  # the proxy opens its clients when it is created
            patch.setattr("tine.proxy.httpx.AsyncClient", Wired)
            app = create_app(app_config, ledger=Ledger())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                return await client.post("/v1/chat/completions", json={"messages": []},
                                         headers={"authorization": "Bearer tenant-token-a"})
    return run(scenario())


def test_a_reply_slower_than_the_old_30_second_default_is_served_by_default(monkeypatch):
    """Long frontier replies took 30 to 53 seconds in the live pilots. A proxy with no
    timeout setting must wait for them; the old default returned 502."""
    monkeypatch.delenv("TINE_TIMEOUT_SECONDS", raising=False)
    assert ProxyConfig().timeout_seconds == 600.0 and ProxyConfig.from_env().timeout_seconds == 600.0
    provider = SlowProvider(seconds=53)
    response = ask_own_client(monkeypatch, provider, config())
    assert response.status_code == 200 and response.json()["choices"][0]["message"]["content"] == "a long answer"
    assert provider.timeouts[0]["read"] == 600.0
    # The setting still applies: at 30 seconds the same reply is a 502, as it was.
    response = ask_own_client(monkeypatch, SlowProvider(seconds=53), config(timeout_seconds=30.0))
    assert response.status_code == 502


def test_an_unreachable_provider_fails_at_the_connect_timeout_not_the_reply_timeout(monkeypatch):
    provider = SlowProvider(seconds=1, connect_seconds=60)
    response = ask_own_client(monkeypatch, provider, config())
    assert response.status_code == 502 and provider.timeouts[0]["connect"] == 10.0
    # A timeout set below it is used for connecting as well.
    provider = SlowProvider(seconds=1)
    ask_own_client(monkeypatch, provider, config(timeout_seconds=4.0))
    assert provider.timeouts[0] == {"connect": 4.0, "read": 4.0, "write": 4.0, "pool": 4.0}


@pytest.mark.parametrize("candidate", [False, True])
def test_streaming_error_body_failure_closes_response_and_uses_fallback(candidate):
    class BrokenBody(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'{"error":'
            raise httpx.ReadError("private upstream connection details")

        async def aclose(self):
            self.closed = True

    broken = BrokenBody()
    attempted = []
    data = b'data: {"choices": []}\n\ndata: [DONE]\n\n'

    def provider(request):
        model = json.loads(request.content)["model"]
        attempted.append(model)
        if candidate and model == "frontier":
            return httpx.Response(200, content=data, headers={"content-type": "text/event-stream"})
        return httpx.Response(503, stream=broken)

    async def scenario():
        ledger = Ledger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(allowed_models={"openai": ["cheap"]}), ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={"model": "cheap" if candidate else "frontier", "messages": [], "stream": True},
                    headers={"authorization": "Bearer tenant-token-a"},
                )
                assert response.status_code == (200 if candidate else 502)
                assert "private upstream" not in response.text
                if candidate:
                    assert response.content == data
                record = ledger.get("a", response.headers["x-tine-request-id"])
                assert record["model"] == "frontier"
                assert record["fallback_used"] == int(candidate)
                assert record["success"] is None
                assert record["cost_usd"] is None
        assert broken.closed
        assert attempted == (["cheap", "frontier"] if candidate else ["frontier"])

    run(scenario())


def test_anthropic_sse_native_headers_and_bytes_pass_through():
    data = b'event: message_start\ndata: {"type":"message_start"}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'

    def provider(request):
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "provider-secret-a"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "authorization" not in request.headers
        assert json.loads(request.content)["model"] == "claude-frontier"
        return httpx.Response(200, content=data, headers={"content-type": "text/event-stream"})

    async def scenario():
        ledger = Ledger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(), ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post("/v1/messages", json={"messages": [], "stream": True, "max_tokens": 10}, headers={"x-api-key": "tenant-token-a"})
                assert response.status_code == 200
                assert response.content == data
                record = ledger.get("a", response.headers["x-tine-request-id"])
                assert record["stream"] == 1
                assert record["success"] is None
                assert record["cost_usd"] is None
                assert record["response_body"] is None

    run(scenario())


STREAM_PRICES = {
    "openai:frontier": {"input_per_million": 1.0, "output_per_million": 2.0},
    "anthropic:claude-frontier": {"input_per_million": 3.0, "output_per_million": 15.0,
                                  "cached_input_per_million": 0.3},
}
ANTHROPIC_STREAM = (
    b'event: message_start\r\ndata: {"type":"message_start","message":{"id":"m","usage":'
    b'{"input_tokens":1000,"cache_read_input_tokens":2000,"output_tokens":1}}}\r\n\r\n'
    b'event: content_block_delta\r\ndata: {"type":"content_block_delta","delta":{"text":"hi \\"usage\\""}}\r\n\r\n'
    b'event: message_delta\r\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":400}}\r\n\r\n'
    b'event: message_stop\r\ndata: {"type":"message_stop"}\r\n\r\n'
)
OPENAI_STREAM = (
    b'data: {"choices":[{"delta":{"content":"hi"}}],"usage":null}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":500}}\n\n'
    b'data: [DONE]\n\n'
)


class PiecewiseStream(httpx.AsyncByteStream):
    """Upstream bytes in pieces of a given size, optionally breaking off after some of them."""

    def __init__(self, data, size, cut_after=None):
        self.data, self.size, self.cut_after = data, size, cut_after

    async def __aiter__(self):
        end = len(self.data) if self.cut_after is None else self.cut_after
        for start in range(0, end, self.size):
            yield self.data[start:min(start + self.size, end)]
        if self.cut_after is not None:
            raise httpx.ReadError("connection reset")

    async def aclose(self):
        pass


def stream_through(provider_name, stream, body=None, **changes):
    """Send one streamed request through the proxy; return what the caller got, the
    ledger row, the body the provider saw and the tenant's savings report."""
    sent = []

    def provider(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    path, auth = (("/v1/chat/completions", {"authorization": "Bearer tenant-token-a"}) if provider_name == "openai"
                  else ("/v1/messages", {"x-api-key": "tenant-token-a"}))

    async def scenario():
        ledger = Ledger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(prices=STREAM_PRICES, **changes), ledger=ledger, client=upstream)
            transport = httpx.ASGITransport(app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://tine") as client:
                response = await client.post(path, json=body or {"messages": [], "stream": True}, headers=auth)
                assert response.status_code == 200
                record = ledger.get("a", response.headers["x-tine-request-id"])
                savings = (await client.get("/v1/savings", headers=auth)).json()
                return response.content, record, sent[0], savings

    return run(scenario())


@pytest.mark.parametrize("size", [1, 7, len(ANTHROPIC_STREAM)])
def test_an_anthropic_stream_is_relayed_unchanged_and_its_usage_is_priced(size):
    content, record, _, savings = stream_through("anthropic", PiecewiseStream(ANTHROPIC_STREAM, size))
    assert content == ANTHROPIC_STREAM
    # Input counts include cache reads; the output is message_delta's, not message_start's 1.
    assert (record["input_tokens"], record["output_tokens"], record["cached_tokens"]) == (3000, 400, 2000)
    assert record["cost_usd"] == pytest.approx((1000 * 3.0 + 2000 * 0.3 + 400 * 15.0) / 1e6)
    assert record["baseline_cost_usd"] is None and record["routing_cost_usd"] is None
    assert record["stream"] == 1 and record["response_body"] is None
    assert savings["unknown_answer_cost_requests"] == 0 and savings["known_answer_cost_total_usd"] == record["cost_usd"]


@pytest.mark.parametrize("size", [1, len(OPENAI_STREAM)])
def test_an_openai_stream_with_usage_is_priced_and_the_request_is_not_changed(size):
    body = {"messages": [], "stream": True, "stream_options": {"include_usage": True}}
    content, record, sent, _ = stream_through("openai", PiecewiseStream(OPENAI_STREAM, size), body)
    assert content == OPENAI_STREAM
    assert sent["stream_options"] == {"include_usage": True}
    assert (record["input_tokens"], record["output_tokens"], record["cached_tokens"]) == (1000, 500, 0)
    assert record["cost_usd"] == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)


def test_a_stream_without_usage_keeps_its_cost_unknown_and_no_option_is_added():
    data = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    content, record, sent, savings = stream_through("openai", PiecewiseStream(data, 5))
    assert content == data
    assert "stream_options" not in sent
    assert record["cost_usd"] is record["input_tokens"] is record["output_tokens"] is record["cached_tokens"] is None
    assert savings["unknown_answer_cost_requests"] == 1


@pytest.mark.parametrize("provider_name,data,marker", [
    ("anthropic", ANTHROPIC_STREAM, b"event: message_delta"),
    ("openai", OPENAI_STREAM, b'data: {"choices":[],"usage"'),
])
def test_a_stream_cut_off_before_its_final_usage_keeps_its_cost_unknown(provider_name, data, marker):
    cut = data.index(marker)
    content, record, _, _ = stream_through(provider_name, PiecewiseStream(data, 9, cut_after=cut))
    assert content == data[:cut]
    assert record["cost_usd"] is None and record["output_tokens"] is None
    # Anthropic's input count arrived before the break and is kept.
    assert record["input_tokens"] == (3000 if provider_name == "anthropic" else None)


def test_a_stream_cut_off_after_its_usage_is_priced():
    cut = ANTHROPIC_STREAM.index(b"event: message_stop")
    _, record, _, _ = stream_through("anthropic", PiecewiseStream(ANTHROPIC_STREAM, 50, cut_after=cut))
    assert record["output_tokens"] == 400 and record["cost_usd"] is not None


def test_a_streamed_cost_is_not_a_measured_baseline_and_the_caller_can_still_supply_one():
    stream = PiecewiseStream(OPENAI_STREAM, 16)
    sent = []

    def provider(request):
        sent.append(request)
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    async def scenario():
        ledger = Ledger()
        auth = {"authorization": "Bearer tenant-token-a"}
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(prices=STREAM_PRICES), ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post("/v1/chat/completions", json={"messages": [], "stream": True},
                                             headers=auth)
                request_id = response.headers["x-tine-request-id"]
                updated = await client.patch(f"/v1/outcomes/{request_id}", headers=auth,
                                             json={"success": True, "routing_cost_usd": 0.0})
                assert updated.status_code == 200
                # Streams are never compared against a measured frontier baseline.
                assert (await client.get("/v1/savings", headers=auth)).json()["comparable_verified_requests"] == 0
                updated = await client.patch(f"/v1/outcomes/{request_id}", headers=auth,
                                             json={"success": True, "baseline_cost_usd": 0.01})
                assert updated.status_code == 200
                savings = (await client.get("/v1/savings", headers=auth)).json()
                assert savings["caller_baseline_requests"] == 1
                assert savings["caller_baseline_savings_usd"] == pytest.approx(0.01 - 0.002)

    run(scenario())


def test_a_fallback_stream_records_its_tokens_but_not_a_cost():
    def provider(request):
        if json.loads(request.content)["model"] == "cheap":
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, stream=PiecewiseStream(OPENAI_STREAM, 32),
                              headers={"content-type": "text/event-stream"})

    async def scenario():
        ledger = Ledger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(prices=STREAM_PRICES, allowed_models={"openai": ["cheap"]}), ledger=ledger,
                             client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post("/v1/chat/completions", json={"model": "cheap", "messages": [],
                                                                         "stream": True},
                                             headers={"authorization": "Bearer tenant-token-a"})
                record = ledger.get("a", response.headers["x-tine-request-id"])
                assert record["fallback_used"] == 1 and record["input_tokens"] == 1000
                assert record["cost_usd"] is None

    run(scenario())


def test_stream_usage_reader_skips_overlong_lines_and_reads_the_last_event_without_a_blank_line():
    reader = _StreamUsage("openai")
    reader.feed(b"data: " + b"x" * (_StreamUsage.MAX_LINE + 10) + b"\n\n")
    reader.feed(b'data: {"usage": {"prompt_tokens": 3, "completion_tokens": 4}}')
    reader.close()
    assert reader.tokens()["input_tokens"] == 3 and reader.tokens()["output_tokens"] == 4
    assert _StreamUsage("anthropic").tokens() is None


def test_late_verified_outcome_enables_exact_cache_only_within_tenant():
    calls = []

    def provider(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 50}})

    async def scenario():
        ledger = Ledger()
        cfg = config(prices={"openai:frontier": {"input_per_million": 1, "output_per_million": 2}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(cfg, ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                body = {"model": "auto", "messages": [], "temperature": 0}
                auth_a = {"authorization": "Bearer tenant-token-a"}
                auth_b = {"authorization": "Bearer tenant-token-b"}
                first = await client.post("/v1/chat/completions", json=body, headers=auth_a)
                first_id = first.headers["x-tine-request-id"]
                assert (await client.patch(f"/v1/outcomes/{first_id}", json={"success": True}, headers=auth_b)).status_code == 404
                assert (await client.patch(f"/v1/outcomes/{first_id}", json={"success": "yes"}, headers=auth_a)).status_code == 422
                assert (await client.patch(f"/v1/outcomes/{first_id}", json={"success": True, "cost_usd": True}, headers=auth_a)).status_code == 422
                assert (await client.patch(f"/v1/outcomes/{first_id}", json={"success": True, "routing_cost_usd": 0.00001}, headers=auth_a)).status_code == 200
                cached = await client.post("/v1/chat/completions", json=body, headers=auth_a)
                assert cached.headers["x-tine-cache"] == "hit"
                cached_id = cached.headers["x-tine-request-id"]
                assert ledger.get("a", cached_id)["cost_usd"] == 0
                assert ledger.get("a", cached_id)["success"] is None
                assert (await client.patch(f"/v1/outcomes/{cached_id}", json={"success": True}, headers=auth_a)).status_code == 200
                incomplete = (await client.get("/v1/savings", headers=auth_a)).json()
                assert incomplete["comparable_verified_requests"] == 1
                assert incomplete["verified_savings_usd"] == pytest.approx(-0.00001)
                assert incomplete["unknown_routing_cost_requests"] == 1
                assert ledger.get("a", cached_id)["routing_cost_usd"] is None
                for request_id in [first_id, cached_id]:
                    assert (await client.patch(f"/v1/outcomes/{request_id}", json={"success": True, "routing_cost_usd": 0.00001}, headers=auth_b)).status_code == 404
                    for invalid in [True, -1, "0"]:
                        assert (await client.patch(f"/v1/outcomes/{request_id}", json={"success": True, "routing_cost_usd": invalid}, headers=auth_a)).status_code == 422
                    assert (await client.patch(f"/v1/outcomes/{request_id}", json={"success": True, "routing_cost_usd": 0.00001}, headers=auth_a)).status_code == 200
                await client.post("/v1/chat/completions", json=body, headers=auth_b)
                await client.post("/v1/chat/completions", json=dict(body, temperature=1), headers=auth_a)
                report = (await client.get("/v1/savings", headers=auth_a)).json()
                assert report["cache_hits"] == 1
                assert report["comparable_verified_requests"] == 2
                assert report["verified_savings_usd"] == pytest.approx(0.00018)
                assert report["verified_routing_cost_usd"] == pytest.approx(0.00002)
                assert len(calls) == 3

    run(scenario())


@pytest.mark.parametrize("changed_headers", [
    {"anthropic-version": "different-version"},
    {"anthropic-beta": "feature-test"},
])
def test_verified_cache_tracks_anthropic_protocol_headers(changed_headers):
    calls = []

    def provider(request):
        calls.append(request)
        return httpx.Response(200, json={"id": f"answer-{len(calls)}", "content": []})

    async def scenario():
        ledger = Ledger()
        auth = {"x-api-key": "tenant-token-a"}
        body = {"messages": [], "max_tokens": 10}
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(), ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                first = await client.post("/v1/messages", json=body, headers=auth)
                ledger.update_outcome("a", first.headers["x-tine-request-id"], True)
                # An explicit default version is equivalent to the implicit one.
                cached = await client.post("/v1/messages", json=body, headers={**auth, "anthropic-version": "2023-06-01"})
                assert cached.headers["x-tine-cache"] == "hit"
                ledger.update_outcome("a", cached.headers["x-tine-request-id"], True)
                changed = await client.post("/v1/messages", json=body, headers={**auth, **changed_headers})
                assert "x-tine-cache" not in changed.headers
                assert changed.json()["id"] == "answer-2"
                for name, value in changed_headers.items():
                    assert calls[-1].headers[name] == value
                # Failure evidence for a different protocol must stay separate.
                ledger.update_outcome("a", changed.headers["x-tine-request-id"], False)
                original = await client.post("/v1/messages", json=body, headers=auth)
                assert original.headers["x-tine-cache"] == "hit"
                assert original.json() == first.json()
        assert len(calls) == 2

    run(scenario())


@pytest.mark.parametrize("provider_name,path", [
    ("openai", "/v1/chat/completions"), ("anthropic", "/v1/messages"),
])
def test_verified_cache_tracks_upstream_endpoint(provider_name, path):
    calls = []

    def provider(request):
        calls.append(request)
        return httpx.Response(200, json={"id": request.url.host})

    async def scenario():
        ledger = Ledger()
        auth = {"authorization": "Bearer tenant-token-a"}
        body = {"messages": [], "max_tokens": 10}
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            for host in ["first.example", "second.example", "first.example"]:
                cfg = config(**{f"{provider_name}_base_url": f"https://{host}"})
                app = create_app(cfg, ledger=ledger, client=upstream)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                    response = await client.post(path, json=body, headers=auth)
                    assert response.json()["id"] == host
                    ledger.update_outcome("a", response.headers["x-tine-request-id"], True)
            assert response.headers["x-tine-cache"] == "hit"
        assert len(calls) == 2

    run(scenario())


def test_configuration_requires_tenant_auth_and_hides_credentials():
    with pytest.raises(ValueError, match="TINE_TENANT_KEYS"):
        create_app(ProxyConfig())
    assert "provider-secret" not in repr(config())
    assert "tenant-token" not in repr(config())


def test_broken_ledger_does_not_discard_provider_answer():
    class BrokenLedger(Ledger):
        def cached(self, *args, **kwargs):
            raise RuntimeError("local path and internal details")

        def record(self, **kwargs):
            raise RuntimeError("local path and internal details")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []}))) as upstream:
            app = create_app(config(), ledger=BrokenLedger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post("/v1/chat/completions", json={"messages": []}, headers={"authorization": "Bearer tenant-token-a"})
                assert response.status_code == 200
                assert response.headers["x-tine-ledger"] == "unavailable"
                assert "internal details" not in response.text

    run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_a_required_ledger_withholds_an_answer_it_cannot_record(stream):
    class BrokenLedger(Ledger):
        def record(self, **kwargs):
            raise RuntimeError("local path and internal details")
    calls = []

    def provider(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "the answer"}}]})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(ledger_required=True), ledger=BrokenLedger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                return await client.post("/v1/chat/completions", json={"messages": [], "stream": stream},
                                         headers={"authorization": "Bearer tenant-token-a"})
    response = run(scenario())
    assert len(calls) == 1 and response.status_code == 503
    assert response.headers["x-tine-ledger"] == "unavailable" and response.headers["x-tine-request-id"]
    assert response.json()["error"]["type"] == "ledger_unavailable"
    assert "the answer" not in response.text and "internal details" not in response.text
    with pytest.raises(ValueError, match="ledger_required"):
        create_app(config(ledger_required="1"), ledger=Ledger())


def test_a_required_ledger_withholds_cached_and_routed_answers_it_cannot_record():
    class FailingLedger(Ledger):
        broken = False

        def record(self, **kwargs):
            if self.broken:
                raise RuntimeError("disk full")
            return super().record(**kwargs)
    policy = {"version": 1, "tenant": "a", "frontier": "frontier",
              "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
              "stages": [{"models": ["cheap"], "check": "none"}]}

    def provider(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    async def scenario(app_config):
        ledger = FailingLedger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(app_config, ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                auth = {"authorization": "Bearer tenant-token-a"}
                body = {"messages": [{"role": "user", "content": "q"}]}
                first = await client.post("/v1/chat/completions", json=body, headers=auth)
                await client.patch(f"/v1/outcomes/{first.headers['x-tine-request-id']}", json={"success": True},
                                   headers=auth)
                ledger.broken = True
                return first, await client.post("/v1/chat/completions", json=body, headers=auth)
    first, cached = run(scenario(config(ledger_required=True)))
    assert first.status_code == 200 and cached.status_code == 503 and "x-tine-cache" not in cached.headers
    routed = config(ledger_required=True, verified_cache=False, route_policies={"a": {"openai": policy}})
    first, second = run(scenario(routed))
    assert first.headers["x-tine-route"] == "stage-0" and second.status_code == 503
    # By default the answer is still served, marked as unrecorded.
    _, served = run(scenario(config()))
    assert served.status_code == 200 and served.headers["x-tine-ledger"] == "unavailable"


def test_the_ledger_requirement_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("TINE_TENANT_KEYS", '{"token": "a"}')
    for name in ("TINE_CHECKS", "TINE_TENANT_CHECKS", "TINE_SHADOW_POLICIES", "TINE_ROUTE_POLICIES"):
        monkeypatch.delenv(name, raising=False)
    for value, expected in (("1", True), ("true", True), ("0", False), ("", False)):
        monkeypatch.setenv("TINE_LEDGER_REQUIRED", value)
        assert ProxyConfig.from_env().ledger_required is expected
    monkeypatch.delenv("TINE_LEDGER_REQUIRED")
    assert ProxyConfig.from_env().ledger_required is False


def test_malformed_usage_never_breaks_provider_answer():
    async def scenario():
        payload = {"choices": [], "usage": {"prompt_tokens_details": [1, 2]}}
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as upstream:
            app = create_app(config(), ledger=Ledger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post("/v1/chat/completions", json={"messages": []}, headers={"authorization": "Bearer tenant-token-a"})
                assert response.status_code == 200
                assert response.json() == payload

    run(scenario())


@pytest.mark.parametrize("details", [None, [], [1, 2], False, 0, "", "invalid"])
def test_malformed_cache_usage_stays_unknown_even_with_valid_token_totals(details):
    async def scenario():
        payload = {"choices": [], "usage": {
            "prompt_tokens": 100, "completion_tokens": 10,
            "prompt_tokens_details": details,
        }}
        cfg = config(prices={"openai:frontier": {
            "input_per_million": 10, "output_per_million": 20,
            "cached_input_per_million": 1,
        }})
        ledger = Ledger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as upstream:
            app = create_app(cfg, ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post("/v1/chat/completions", json={"messages": []},
                                             headers={"authorization": "Bearer tenant-token-a"})
                assert response.status_code == 200 and response.json() == payload
                assert response.headers["x-tine-ledger"] == "recorded"
                record = ledger.get("a", response.headers["x-tine-request-id"])
                assert record["input_tokens"] == 100 and record["output_tokens"] == 10
                assert record["cached_tokens"] is None
                assert record["cost_usd"] is None and record["baseline_cost_usd"] is None
                ledger.update_outcome("a", record["request_id"], True)
                assert ledger.report("a")["comparable_verified_requests"] == 0

    run(scenario())


@pytest.mark.parametrize("extra,cached,expected_cost", [
    ({}, 0, 0.0012),
    ({"prompt_tokens_details": {}}, 0, 0.0012),
    ({"prompt_tokens_details": {"cached_tokens": 0}}, 0, 0.0012),
    ({"prompt_tokens_details": {"cached_tokens": 40}}, 40, 0.00084),
])
def test_valid_cache_usage_preserves_configured_costs(extra, cached, expected_cost):
    tokens = _usage("openai", {"usage": {"prompt_tokens": 100, "completion_tokens": 10, **extra}})
    cfg = config(prices={"openai:frontier": {
        "input_per_million": 10, "output_per_million": 20, "cached_input_per_million": 1,
    }})
    assert tokens["cached_tokens"] == cached
    assert _cost(cfg, "openai", "frontier", tokens) == pytest.approx(expected_cost)


@pytest.mark.parametrize("path,usage", [
    ("/v1/chat/completions", {"prompt_tokens": 10**400, "completion_tokens": 1}),
    ("/v1/messages", {"input_tokens": 2**63, "output_tokens": 1}),
    ("/v1/messages", {"input_tokens": 2**63 - 1, "cache_read_input_tokens": 1, "output_tokens": 1}),
])
def test_oversized_usage_preserves_answer_and_unknown_accounting(path, usage):
    async def scenario():
        payload = {"content": "answer", "usage": usage}
        cfg = config(prices={
            "openai:frontier": {"input_per_million": 1, "output_per_million": 2},
            "anthropic:claude-frontier": {"input_per_million": 1, "output_per_million": 2},
        })
        ledger = Ledger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as upstream:
            app = create_app(cfg, ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post(path, json={"messages": []}, headers={"authorization": "Bearer tenant-token-a"})
                assert response.status_code == 200
                assert response.json() == payload
                assert response.headers["x-tine-ledger"] == "recorded"
                record = ledger.get("a", response.headers["x-tine-request-id"])
                assert record["input_tokens"] is None
                assert record["cost_usd"] is None
                assert record["success"] is None
    run(scenario())


def test_invalid_unicode_request_is_rejected_before_calling_provider():
    calls = []

    def provider(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": []})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(), ledger=Ledger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                for text in ["\ud800", "\udfff"]:
                    response = await client.post(
                        "/v1/chat/completions", content=json.dumps({"messages": [{"role": "user", "content": text}]}),
                        headers={"authorization": "Bearer tenant-token-a", "content-type": "application/json"},
                    )
                    assert response.status_code == 400
                valid = await client.post(
                    "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello 🌍"}]},
                    headers={"authorization": "Bearer tenant-token-a"},
                )
                assert valid.status_code == 200
        assert len(calls) == 1
    run(scenario())


@pytest.mark.parametrize("header", [b"anthropic-version", b"anthropic-beta"])
def test_invalid_protocol_header_returns_client_error(header):
    def provider(request):
        pytest.fail("Invalid protocol header reached provider")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(), ledger=Ledger(), client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                response = await client.post("/v1/messages", json={"messages": []},
                                             headers=[(b"x-api-key", b"tenant-token-a"), (header, b"\xff")])
                assert response.status_code == 400
    run(scenario())


def ledger_rows(ledger):
    return ledger._db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]


def test_request_limits_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("TINE_TENANT_KEYS", '{"token": "a"}')
    for name in ("TINE_CHECKS", "TINE_TENANT_CHECKS", "TINE_SHADOW_POLICIES", "TINE_ROUTE_POLICIES",
                 "TINE_MAX_BODY_BYTES", "TINE_DENIED_FIELDS", "TINE_ANTHROPIC_BETA_ALLOWLIST"):
        monkeypatch.delenv(name, raising=False)
    default = ProxyConfig.from_env()
    assert default.max_body_bytes == 10_485_760
    assert default.denied_fields == {} and default.anthropic_beta_allowlist is None
    monkeypatch.setenv("TINE_MAX_BODY_BYTES", "2048")
    monkeypatch.setenv("TINE_DENIED_FIELDS", '{"openai": ["store"], "anthropic": ["mcp_servers", "container"]}')
    monkeypatch.setenv("TINE_ANTHROPIC_BETA_ALLOWLIST", " beta-a , beta-b,")
    loaded = ProxyConfig.from_env()
    loaded.validate()
    assert loaded.max_body_bytes == 2048
    assert loaded.denied_fields == {"openai": ["store"], "anthropic": ["mcp_servers", "container"]}
    assert loaded.anthropic_beta_allowlist == ["beta-a", "beta-b"]
    monkeypatch.setenv("TINE_ANTHROPIC_BETA_ALLOWLIST", "")
    assert ProxyConfig.from_env().anthropic_beta_allowlist is None


@pytest.mark.parametrize("chunked", [False, True])
def test_an_oversized_body_gets_413_before_the_provider_or_the_ledger(chunked):
    calls = []

    def provider(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": []})

    def payload(size):
        body = json.dumps({"messages": [{"role": "user", "content": ""}]}).encode()
        return body.replace(b'""', b'"' + b"x" * (size - len(body)) + b'"')

    async def chunks(data):
        for start in range(0, len(data), 100):
            yield data[start:start + 100]

    async def scenario():
        ledger = Ledger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(max_body_bytes=1000), ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                for size, status in ((1001, 413), (1000, 200)):
                    data = payload(size)
                    assert len(data) == size
                    response = await client.post(
                        "/v1/chat/completions", content=chunks(data) if chunked else data,
                        headers={"authorization": "Bearer tenant-token-a", "content-type": "application/json"},
                    )
                    assert response.status_code == status
                    if status == 413:
                        assert "1000 bytes" in response.json()["detail"]
                        assert not calls and ledger_rows(ledger) == 0
        assert len(calls) == 1 and ledger_rows(ledger) == 1

    run(scenario())


def test_denied_fields_get_400_per_provider_and_are_not_recorded():
    calls = []

    def provider(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [], "content": []})

    async def scenario():
        ledger = Ledger()
        cfg = config(denied_fields={"openai": ["store"], "anthropic": ["mcp_servers", "container"]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(cfg, ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                openai = {"authorization": "Bearer tenant-token-a"}
                anthropic = {"x-api-key": "tenant-token-a"}
                for path, body, headers in (
                    ("/v1/chat/completions", {"messages": [], "store": False}, openai),
                    ("/v1/messages", {"messages": [], "mcp_servers": []}, anthropic),
                    ("/v1/messages", {"messages": [], "container": "c", "mcp_servers": [{}]}, anthropic),
                ):
                    response = await client.post(path, json=body, headers=headers)
                    assert response.status_code == 400
                    assert all(name in response.json()["detail"] for name in body if name != "messages")
                assert not calls and ledger_rows(ledger) == 0
                # A field denied for one provider is still forwarded to the other.
                response = await client.post("/v1/messages", json={"messages": [], "store": True}, headers=anthropic)
                assert response.status_code == 200
                response = await client.post("/v1/chat/completions", json={"messages": [], "container": "c"},
                                             headers=openai)
                assert response.status_code == 200
        assert [body.get("store", body.get("container")) for body in calls] == [True, "c"]
        assert ledger_rows(ledger) == 2

    run(scenario())


def test_anthropic_beta_allowlist_refuses_other_values_and_unset_forwards_all():
    seen = []

    def provider(request):
        seen.append(request.headers.get("anthropic-beta"))
        return httpx.Response(200, json={"content": []})

    async def scenario(allowlist, values):
        ledger = Ledger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(config(anthropic_beta_allowlist=allowlist), ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                statuses = []
                for value in values:
                    headers = {"x-api-key": "tenant-token-a"}
                    if value is not None:
                        headers["anthropic-beta"] = value
                    response = await client.post("/v1/messages", json={"messages": []}, headers=headers)
                    statuses.append(response.status_code)
                    if response.status_code == 400:
                        assert "beta-c" in response.json()["detail"]
                        assert "beta-a" not in response.json()["detail"]
                return statuses, ledger_rows(ledger)

    values = [None, "beta-a", "beta-a, beta-b", "beta-a,beta-c", "beta-c"]
    assert run(scenario(["beta-a", "beta-b"], values)) == ([200, 200, 200, 400, 400], 3)
    assert seen == [None, "beta-a", "beta-a, beta-b"]
    seen.clear()
    assert run(scenario(None, values)) == ([200] * 5, 5)
    assert seen == values
