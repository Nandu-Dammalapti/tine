import pytest

from tine.ledger import Ledger, request_fingerprint


def row(ledger, tenant="a", **overrides):
    values = dict(
        tenant=tenant,
        provider="openai",
        model="frontier",
        request_body={"model": "frontier", "messages": [{"role": "user", "content": "hello"}]},
        response_body={"choices": [{"message": {"content": "hello"}}]},
        upstream_status=200,
    )
    values.update(overrides)
    return ledger.record(**values)


def test_unknown_outcomes_and_costs_are_not_success_or_savings():
    ledger = Ledger()
    request_id = row(ledger)
    report = ledger.report("a")
    assert report["requests"] == 1
    assert report["pending_outcomes"] == 1
    assert report["unknown_cost_requests"] == 1
    assert report["comparable_verified_requests"] == 0
    assert report["savings_fraction"] is None
    assert ledger.get("b", request_id) is None
    assert not ledger.update_outcome("b", request_id, True, cost_usd=0)
    assert ledger.get("a", request_id)["success"] is None


def test_late_outcomes_cost_updates_and_tenant_reports():
    ledger = Ledger()
    request_id = row(ledger)
    row(ledger, tenant="b", success=True, cost_usd=900, baseline_cost_usd=1000)
    assert ledger.update_outcome("a", request_id, True, cost_usd=0.25, baseline_cost_usd=1.0)
    report = ledger.report("a")
    assert report["requests"] == 1
    assert report["verified_savings_usd"] == 0.75
    assert report["savings_fraction"] == 0.75
    assert ledger.update_outcome("a", request_id, False)
    assert ledger.report("a")["comparable_verified_requests"] == 0


def test_cache_needs_verified_outcome_and_exact_tenant_request():
    ledger = Ledger()
    body = {"model": "frontier", "messages": [], "temperature": 0}
    request_id = row(ledger, request_body=body)
    assert ledger.cached("a", "openai", body) is None
    ledger.update_outcome("a", request_id, True)
    assert ledger.cached("a", "openai", body)["request_id"] == request_id
    assert ledger.cached("b", "openai", body) is None
    assert ledger.cached("a", "anthropic", body) is None
    assert ledger.cached("a", "openai", dict(body, temperature=1)) is None
    assert ledger.cached("a", "openai", dict(body, model="cheap")) is None
    ledger.update_outcome("a", request_id, False)
    assert ledger.cached("a", "openai", body) is None


@pytest.mark.parametrize("invalid", [True, -1, float("nan"), float("inf"), "1"])
def test_cost_validation(invalid):
    ledger = Ledger()
    with pytest.raises(ValueError):
        row(ledger, cost_usd=invalid)
    request_id = row(ledger)
    with pytest.raises(ValueError):
        ledger.update_outcome("a", request_id, True, cost_usd=invalid)


def test_nonboolean_outcomes_and_stream_cache_rejected():
    ledger = Ledger()
    with pytest.raises(ValueError):
        row(ledger, success=1)
    request_id = row(ledger, stream=True, success=True)
    assert ledger.cached("a", "openai", ledger.get("a", request_id)["request_body"]) is None
    assert request_fingerprint("openai", {"a": 1, "b": 2}) == request_fingerprint("openai", {"b": 2, "a": 1})


def test_cache_uses_latest_resolved_outcome_and_late_reversal():
    ledger = Ledger()
    body = {"model": "frontier", "messages": []}
    first = row(ledger, request_body=body, success=True)
    assert ledger.cached("a", "openai", body)["request_id"] == first
    second = row(ledger, request_body=body, success=False)
    assert ledger.cached("a", "openai", body) is None
    ledger.update_outcome("a", second, True)
    assert ledger.cached("a", "openai", body)["request_id"] == second
    # A late correction to an older request is newer outcome evidence.
    ledger.update_outcome("a", first, False)
    assert ledger.cached("a", "openai", body) is None


def test_context_cache_keys_survive_reopen_and_exclude_legacy_entries(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    body = {"model": "frontier", "messages": []}
    context = {"url": "https://provider.example/v1/chat/completions", "headers": {}}
    ledger = Ledger(path)
    legacy_id = row(ledger, request_body=body, success=True)
    ledger.close()

    ledger = Ledger(path)
    assert ledger.cached("a", "openai", body)["request_id"] == legacy_id
    assert ledger.cached("a", "openai", body, cache_context=context) is None
    scoped_id = row(ledger, request_body=body, success=True, cache_context=context)
    ledger.close()

    ledger = Ledger(path)
    assert ledger.cached("a", "openai", body, cache_context=context)["request_id"] == scoped_id
    assert ledger.cached("a", "openai", body)["request_id"] == legacy_id
    assert ledger.cached("b", "openai", body, cache_context=context) is None
    ledger.update_outcome("a", legacy_id, False)
    assert ledger.cached("a", "openai", body, cache_context=context)["request_id"] == scoped_id
    assert ledger.report("a")["requests"] == 2
    ledger.close()


def test_known_cost_per_success_includes_failed_request_spend():
    ledger = Ledger()
    row(ledger, success=True, cost_usd=1, baseline_cost_usd=2)
    row(ledger, success=False, cost_usd=3)
    report = ledger.report("a")
    assert report["known_cost_per_verified_success_usd"] == 4
    assert report["verified_savings_usd"] == 1
    assert "not a billing amount" in report["savings_scope"]
