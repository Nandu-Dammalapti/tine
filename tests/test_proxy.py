import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from tine.ledger import Ledger
from tine.proxy import ProxyConfig, _cost, create_app


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
