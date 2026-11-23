import asyncio
import json
import logging

import httpx
import pytest

from tine.ledger import Ledger
from tine.policy import expired, expiry, validate_policy
from tine.proxy import ProxyConfig, create_app
from tine.shadow import ShadowStore, WHOLE_REPLY_PATTERN

AGREE = {"version": 1, "tenant": "a", "frontier": "frontier",
         "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03},
         "stages": [{"models": ["small-a", "small-b"], "check": "agreement"}],
         "provenance": {"created": "2026-11-01T00:00:00+00:00"}}
PRICES = {f"openai:{name}": {"input_per_million": rate, "output_per_million": rate}
          for name, rate in (("frontier", 100.0), ("small-a", 1.0), ("small-b", 2.0))}
METRICS_TOKEN = "metrics-token-0123456789"


def config(**changes):
    values = dict(tenant_keys={"token-a": "a"}, openai_api_key="secret", openai_frontier_model="frontier",
                  prices=PRICES, verified_cache=False, answer_pattern=WHOLE_REPLY_PATTERN)
    values.update(changes)
    return ProxyConfig(**values)


def provider(seen):
    def handler(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}})
    return handler


def session(app_config, handler, steps):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
            ledger = Ledger()
            app = create_app(app_config, ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                return await steps(client, ledger)
    return asyncio.run(scenario())


def ask(client):
    return client.post("/v1/chat/completions", headers={"authorization": "Bearer token-a"},
                       json={"model": "frontier", "messages": [{"role": "user", "content": "q"}]})


def test_metrics_are_off_without_a_token_and_answer_only_the_metrics_token():
    async def steps(client, _):
        await ask(client)
        return await client.get("/metrics")
    assert session(config(), provider([]), steps).status_code == 404

    async def steps(client, _):
        await ask(client)
        await ask(client)
        tenant = await client.get("/metrics", headers={"authorization": "Bearer token-a"})
        good = await client.get("/metrics", headers={"authorization": f"Bearer {METRICS_TOKEN}"})
        return tenant, good
    tenant, good = session(config(metrics_token=METRICS_TOKEN, route_policies={"a": {"openai": AGREE}}),
                           provider([]), steps)
    assert tenant.status_code == 401 and good.status_code == 200
    text = good.text
    assert 'tine_requests_total{provider="openai",route="stage-0",status="2xx",tenant="a"} 2' in text
    assert 'tine_request_seconds_count{provider="openai",route="stage-0"} 2' in text
    assert 'tine_policy_expired{mode="route",provider="openai",tenant="a"} 0' in text
    assert "token-a" not in text and "secret" not in text


@pytest.mark.parametrize("token", ["short", "token-a"])
def test_a_weak_or_reused_metrics_token_is_refused(token):
    with pytest.raises(ValueError, match="TINE_METRICS_TOKEN"):
        create_app(config(metrics_token=token), ledger=Ledger())


def test_each_request_is_logged_once_without_its_text(caplog):
    async def steps(client, _):
        return await ask(client)
    with caplog.at_level(logging.INFO, logger="tine.requests"):
        response = session(config(), provider([]), steps)
    (record,) = [record for record in caplog.records if record.name == "tine.requests"]
    assert record.tine["request_id"] == response.headers["x-tine-request-id"]
    assert record.tine["tenant"] == "a" and record.tine["status"] == 200 and record.tine["model"] == "frontier"
    assert '"q"' not in json.dumps(record.tine)


def test_an_expired_certificate_sends_requests_to_the_frontier_and_says_why():
    now = expiry(AGREE, 30)
    assert now.isoformat() == "2026-12-01T00:00:00+00:00"
    assert expired(AGREE, 30, now) is not None and expired(AGREE, 30, now.replace(day=30, month=11)) is None
    assert expired({**AGREE, "provenance": {}}, 30) is not None  # an age that cannot be known has run out
    assert expired({**AGREE, "expires": "2999-01-01"}) is None and expired({**AGREE, "expires": "2000-01-01"})
    with pytest.raises(ValueError, match="expires"):
        validate_policy({**AGREE, "expires": "soon"})
    seen = []

    async def steps(client, ledger):
        response = await ask(client)
        health = await client.get("/health")
        return response, ledger.get("a", response.headers["x-tine-request-id"]), health.json()
    response, row, health = session(config(route_policies={"a": {"openai": AGREE}}, policy_max_age_days=1),
                                    provider(seen), steps)
    assert seen == ["frontier"] and response.headers["x-tine-route-skipped"] == "expired"
    assert row["route_skip_reason"].startswith("expired") and row["routing_cost_usd"] == 0.0
    assert health["expired_policies"] == 1


def test_a_drift_check_change_is_stored_once_and_kept_after_it_goes_away():
    store = ShadowStore(Ledger())
    warn = {"drift_policy": "openai", "drift_check": "warning", "drift_requests": 400, "drift_compared": 300,
            "drift_differing": 40, "drift_epsilon": 0.04, "drift_policies": None}
    assert len(store.note_drift("a", warn)) == 1
    assert store.note_drift("a", warn) == []
    assert len(store.note_drift("a", {**warn, "drift_check": "no clear sign"})) == 1
    history = store.drift_history("a")
    assert [event["status"] for event in history] == ["warning", "no clear sign"]
    assert history[0]["drift_differing"] == 40 and store.drift_history("b") == []


def test_the_log_format_is_text_or_json(monkeypatch):
    from tine.proxy import JsonLogFormatter, log_formatter
    record = logging.LogRecord("tine.requests", logging.INFO, "x", 1, "request", None, None)
    record.tine = {"tenant": "a", "status": 200}
    monkeypatch.setenv("TINE_LOG_FORMAT", "json")
    formatter = log_formatter()
    assert isinstance(formatter, JsonLogFormatter)
    line = json.loads(formatter.format(record))
    assert line["message"] == "request" and line["tenant"] == "a" and line["status"] == 200
    monkeypatch.setenv("TINE_LOG_FORMAT", "text")
    assert log_formatter().format(record).endswith('request {"tenant": "a", "status": 200}')
    monkeypatch.setenv("TINE_LOG_FORMAT", "xml")
    with pytest.raises(ValueError, match="TINE_LOG_FORMAT"):
        log_formatter()
