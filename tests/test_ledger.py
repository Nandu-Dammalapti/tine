import pytest
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

from tine.ledger import (DEFAULT_LEDGER_PATH, HOLDOUT_MIN_REQUESTS, HoldoutCostError, Ledger, backup_ledger,
                         holdout_estimate, request_fingerprint)


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


def test_usage_learned_after_recording_fills_only_empty_fields_and_stays_in_the_tenant():
    ledger = Ledger()
    streamed = row(ledger, stream=True, response_body=None)
    supplied = row(ledger, stream=True, response_body=None)
    assert ledger.update_outcome("a", supplied, True, cost_usd=0.5)
    assert not ledger.fill_usage("b", streamed, input_tokens=1, cost_usd=0.1)
    for request_id in (streamed, supplied):
        assert ledger.fill_usage("a", request_id, input_tokens=10, output_tokens=20, cached_tokens=0, cost_usd=0.1)
    filled = ledger.get("a", streamed)
    assert (filled["input_tokens"], filled["output_tokens"], filled["cached_tokens"], filled["cost_usd"]) == (
        10, 20, 0, 0.1)
    assert filled["baseline_cost_usd"] is filled["baseline_source"] is filled["success"] is None
    # A cost the caller supplied first is kept, and no outcome event is added.
    assert ledger.get("a", supplied)["cost_usd"] == 0.5 and len(ledger.outcome_events("a", supplied)) == 1
    for bad in ({"input_tokens": -1}, {"output_tokens": True}, {"cost_usd": float("nan")}):
        with pytest.raises(ValueError):
            ledger.fill_usage("a", streamed, **bad)


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
    assert ledger.update_outcome("a", request_id, True, cost_usd=0.25, routing_cost_usd=0.0, baseline_cost_usd=1.0)
    report = ledger.report("a")
    assert report["requests"] == 1
    # The caller supplied the baseline: its saving is reported, apart from verified savings.
    assert report["verified_savings_usd"] is None and report["comparable_verified_requests"] == 0
    assert report["caller_baseline_requests"] == 1 and report["caller_baseline_savings_usd"] == 0.75
    assert ledger.update_outcome("a", request_id, False)
    assert ledger.report("a")["caller_baseline_requests"] == 0
    measured = row(ledger, success=True, cost_usd=0.25, routing_cost_usd=0.0, baseline_cost_usd=1.0)
    report = ledger.report("a")
    assert ledger.get("a", measured)["baseline_source"] == "proxy_measured"
    assert report["verified_savings_usd"] == 0.75 and report["savings_fraction"] == 0.75


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


@pytest.mark.parametrize("field", ["cost_usd", "routing_cost_usd", "baseline_cost_usd"])
@pytest.mark.parametrize("invalid", [True, -1, float("nan"), float("inf"), "1"])
def test_cost_validation(invalid, field):
    ledger = Ledger()
    with pytest.raises(ValueError):
        row(ledger, **{field: invalid})
    request_id = row(ledger)
    with pytest.raises(ValueError):
        ledger.update_outcome("a", request_id, True, **{field: invalid})


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
    row(ledger, success=True, cost_usd=1, routing_cost_usd=0, baseline_cost_usd=2)
    row(ledger, success=False, cost_usd=3, routing_cost_usd=0)
    report = ledger.report("a")
    assert report["known_cost_per_verified_success_usd"] == 4
    assert report["verified_savings_usd"] == 1
    assert "not a billing amount" in report["savings_scope"]


def test_savings_scope_says_live_routing_saving_needs_a_frontier_holdout():
    report = Ledger().report("a")
    assert "Under live routing" in report["savings_scope"]
    assert "not the policy's saving" in report["savings_scope"]
    assert "always sent to the frontier: with TINE_AUDIT_RATE set, holdout_estimate" in report["savings_scope"]


def test_routing_costs_can_exceed_apparent_answer_savings():
    ledger = Ledger()
    row(ledger, success=True, cost_usd=0.25, routing_cost_usd=0.9, baseline_cost_usd=1.0)
    row(ledger, tenant="b", success=True, cost_usd=100, routing_cost_usd=500, baseline_cost_usd=1000)
    report = ledger.report("a")
    assert report["verified_answer_cost_usd"] == 0.25
    assert report["verified_routing_cost_usd"] == 0.9
    assert report["verified_cost_usd"] == pytest.approx(1.15)
    assert report["verified_savings_usd"] == pytest.approx(-0.15)
    assert report["savings_fraction"] == pytest.approx(-0.15)


def test_unknown_routing_costs_require_explicit_measurement_for_net_savings():
    ledger = Ledger()
    request_id = row(ledger, success=True, cost_usd=0.25, baseline_cost_usd=1)
    report = ledger.report("a")
    assert report["unknown_routing_cost_requests"] == report["unknown_cost_requests"] == 1
    assert report["unknown_answer_cost_requests"] == 0
    assert report["comparable_verified_requests"] == 0
    assert report["verified_savings_usd"] is None
    assert report["known_cost_per_verified_success_usd"] is None
    assert not ledger.update_outcome("b", request_id, True, routing_cost_usd=0)
    assert ledger.get("a", request_id)["routing_cost_usd"] is None
    assert ledger.update_outcome("a", request_id, True, routing_cost_usd=0)
    assert ledger.report("a")["verified_savings_usd"] == 0.75
    ledger.update_outcome("a", request_id, True)
    assert ledger.get("a", request_id)["routing_cost_usd"] == 0
    ledger.update_outcome("a", request_id, True, routing_cost_usd=None)
    assert ledger.report("a")["verified_savings_usd"] is None


def test_cost_per_success_includes_routing_on_failed_and_pending_attempts():
    ledger = Ledger()
    row(ledger, success=True, cost_usd=1, routing_cost_usd=0.5, baseline_cost_usd=2)
    row(ledger, success=False, cost_usd=3, routing_cost_usd=0.25)
    row(ledger, cost_usd=2, routing_cost_usd=0.75)
    report = ledger.report("a")
    assert report["known_answer_cost_total_usd"] == 6
    assert report["known_routing_cost_total_usd"] == 1.5
    assert report["known_cost_total_usd"] == report["known_cost_per_verified_success_usd"] == 7.5
    assert report["verified_savings_usd"] == 0.5
    row(ledger, cost_usd=None, routing_cost_usd=0.1)
    report = ledger.report("a")
    assert report["unknown_answer_cost_requests"] == 1
    assert report["known_routing_cost_total_usd"] == pytest.approx(1.6)
    assert report["known_cost_per_verified_success_usd"] is None


def test_routing_cost_schema_upgrade_preserves_unknowns_and_updates(tmp_path):
    import sqlite3

    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    request_id = row(ledger, success=True, cost_usd=0.25, baseline_cost_usd=1)
    ledger.close()
    with sqlite3.connect(path) as database:
        database.execute("ALTER TABLE requests DROP COLUMN routing_cost_usd")
    migrated = Ledger(path)
    assert migrated.get("a", request_id)["cost_usd"] == 0.25
    assert migrated.get("a", request_id)["routing_cost_usd"] is None
    assert migrated.report("a")["comparable_verified_requests"] == 0
    migrated.update_outcome("a", request_id, True, routing_cost_usd=0.1)
    migrated.close()
    reopened = Ledger(path)
    assert reopened.report("a")["verified_savings_usd"] == pytest.approx(0.65)
    reopened.close()


def test_route_skip_reason_column_is_added_to_an_older_ledger(tmp_path):
    import sqlite3

    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    request_id = row(ledger, success=True, cost_usd=0.25)
    ledger.close()
    with sqlite3.connect(path) as database:
        database.execute("ALTER TABLE requests DROP COLUMN route_skip_reason")
    migrated = Ledger(path)
    assert migrated.get("a", request_id)["route_skip_reason"] is None
    skipped = row(migrated, route_skip_reason="tools: why")
    assert migrated.get("a", skipped)["route_skip_reason"] == "tools: why"
    with pytest.raises(ValueError, match="route_skip_reason"):
        row(migrated, route_skip_reason="")
    migrated.close()


def test_net_cost_overflow_stays_unknown_and_report_remains_json_serializable():
    import json

    ledger = Ledger()
    row(ledger, success=True, cost_usd=1e308, routing_cost_usd=1e308, baseline_cost_usd=1)
    report = ledger.report("a")
    assert report["known_cost_total_usd"] is None
    assert report["verified_cost_usd"] is None
    assert report["verified_savings_usd"] is None
    assert report["savings_fraction"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("later_hour", [9, 10])
def test_latest_outcome_wins_when_wall_clock_repeats_or_moves_backwards(later_hour):
    ledger = Ledger()
    body = {"model": "frontier", "messages": []}
    with patch("tine.ledger.datetime") as clock:
        clock.now.return_value = datetime(2026, 11, 9, 10, tzinfo=timezone.utc)
        first = row(ledger, request_body=body, success=True)
        clock.now.return_value = datetime(2026, 11, 9, later_hour, tzinfo=timezone.utc)
        second = row(ledger, request_body=body, success=False)
        assert ledger.cached("a", "openai", body) is None
        ledger.update_outcome("a", first, True)
        assert ledger.cached("a", "openai", body)["request_id"] == first
        ledger.update_outcome("a", second, False)
        assert ledger.cached("a", "openai", body) is None
    ledger.close()


def test_outcome_order_is_shared_across_connections_and_survives_reopen(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    first, second = Ledger(path), Ledger(path)
    body = {"model": "frontier", "messages": []}
    ids = [row(first, request_body=body, success=True), row(second, request_body=body, success=True)]
    barrier = Barrier(2)

    def update(store, request_id, success):
        barrier.wait(timeout=5)
        store.update_outcome("a", request_id, success)

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(update, first, ids[0], False), pool.submit(update, second, ids[1], True)]
        for job in jobs:
            job.result(timeout=10)
    records = [first.get("a", request_id) for request_id in ids]
    assert len({record["outcome_sequence"] for record in records}) == 2
    latest = max(records, key=lambda record: record["outcome_sequence"])
    first.close()
    second.close()
    reopened = Ledger(path)
    cached = reopened.cached("a", "openai", body)
    assert (cached["request_id"] if cached else None) == (latest["request_id"] if latest["success"] else None)
    assert reopened.cached("b", "openai", body) is None
    reopened.close()


def test_legacy_cache_migration_preserves_rows_and_orders_new_evidence_first(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    body = {"model": "frontier", "messages": []}
    ledger = Ledger(path)
    with patch("tine.ledger.datetime") as clock:
        clock.now.return_value = datetime(2030, 1, 1, tzinfo=timezone.utc)
        request_id = row(ledger, request_body=body, success=True, cost_usd=0.5)
    ledger.close()
    # Reconstruct the original ledger schema, before timestamp/sequence ordering.
    with sqlite3.connect(path) as database:
        columns = [column[1] for column in database.execute("PRAGMA table_info(requests)")
                   if column[1] not in {"outcome_updated_at", "outcome_sequence"}]
        database.execute(f"CREATE TABLE legacy_requests AS SELECT {', '.join(columns)} FROM requests")
        database.execute("DROP TABLE requests")
        database.execute("ALTER TABLE legacy_requests RENAME TO requests")
        database.execute("DROP TABLE IF EXISTS ledger_state")
    migrated = Ledger(path)
    assert migrated.cached("a", "openai", body)["request_id"] == request_id
    assert migrated.get("a", request_id)["cost_usd"] == 0.5
    with patch("tine.ledger.datetime") as clock:
        clock.now.return_value = datetime(2020, 1, 1, tzinfo=timezone.utc)
        row(migrated, request_body=body, success=False)
    assert migrated.cached("a", "openai", body) is None
    assert migrated.report("a")["requests"] == 2
    migrated.close()


def test_an_estimated_baseline_is_stored_as_one_and_reported_apart_from_measured_savings():
    ledger = Ledger()
    measured = row(ledger, success=True, cost_usd=1.0, routing_cost_usd=0.5, baseline_cost_usd=10.0)
    estimated = row(ledger, model="cheap", success=True, cost_usd=1.0, routing_cost_usd=1.0, baseline_cost_usd=20.0,
                    baseline_estimated=True)
    assert ledger.get("a", measured)["baseline_estimated"] == 0 and ledger.get("a", estimated)["baseline_estimated"] == 1
    report = ledger.report("a")
    assert report["comparable_verified_requests"] == 1 and report["verified_baseline_cost_usd"] == 10.0
    assert report["verified_savings_usd"] == 8.5 and report["savings_fraction"] == 0.85
    assert report["estimated_baseline_requests"] == 1 and report["estimated_baseline_savings_usd"] == 18.0
    assert "estimate" in report["estimated_baseline_scope"] and "estimated" in report["savings_scope"]
    # Leaving the baseline alone keeps it an estimate; a baseline the caller supplies is the caller's.
    assert ledger.update_outcome("a", estimated, True, cost_usd=2.0)
    assert ledger.get("a", estimated)["baseline_estimated"] == 1
    assert ledger.get("a", estimated)["baseline_source"] == "estimated"
    assert ledger.update_outcome("a", estimated, True, baseline_cost_usd=30.0)
    report = ledger.report("a")
    assert ledger.get("a", estimated)["baseline_estimated"] == 0 and report["estimated_baseline_requests"] == 0
    assert ledger.get("a", estimated)["baseline_source"] == "caller"
    assert report["comparable_verified_requests"] == 1 and report["estimated_baseline_savings_usd"] is None
    assert report["caller_baseline_requests"] == 1 and report["caller_baseline_savings_usd"] == 27.0
    with pytest.raises(ValueError):
        row(ledger, baseline_estimated="yes")
    with pytest.raises(ValueError):
        row(ledger, baseline_estimated=True)  # an estimate of nothing


def test_a_ledger_written_before_estimates_were_marked_still_opens_and_reports(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite3"
    ledger = Ledger(path)
    row(ledger, success=True, cost_usd=1.0, routing_cost_usd=0.0, baseline_cost_usd=4.0)
    ledger._db.execute("ALTER TABLE requests DROP COLUMN baseline_estimated")
    ledger._db.commit()
    ledger.close()
    # Read without upgrading, as the report script does for a read-only file.
    old = Ledger.__new__(Ledger)
    old._lock, old._db = __import__("threading").RLock(), sqlite3.connect(str(path))
    old._db.row_factory = sqlite3.Row
    assert old.report("a")["verified_savings_usd"] == 3.0 and old.report("a")["estimated_baseline_requests"] == 0
    old.close()
    upgraded = Ledger(path)
    assert upgraded.report("a")["verified_savings_usd"] == 3.0
    assert row(upgraded, success=True, cost_usd=1.0, routing_cost_usd=0.0, baseline_cost_usd=4.0,
               baseline_estimated=True)
    assert upgraded.report("a")["estimated_baseline_requests"] == 1
    upgraded.close()


def test_a_caller_cannot_replace_a_baseline_the_proxy_measured():
    from tine.ledger import MeasuredBaselineError
    ledger = Ledger()
    measured = row(ledger, cost_usd=1.0, routing_cost_usd=0.0, baseline_cost_usd=1.0)
    assert ledger.get("a", measured)["baseline_source"] == "proxy_measured"
    for replacement in (0.02, None):
        with pytest.raises(MeasuredBaselineError):
            ledger.update_outcome("a", measured, True, baseline_cost_usd=replacement)
    # Nothing of a refused update is kept, not even its outcome.
    record = ledger.get("a", measured)
    assert (record["success"], record["baseline_cost_usd"], record["baseline_source"]) == (None, 1.0, "proxy_measured")
    assert ledger.outcome_events("a", measured) == []
    assert ledger.update_outcome("a", measured, True)
    assert ledger.report("a")["verified_savings_usd"] == 0.0
    # A request with no baseline yet takes the caller's, and the caller may correct it.
    unmeasured = row(ledger, cost_usd=1.0, routing_cost_usd=0.0)
    assert ledger.update_outcome("a", unmeasured, True, baseline_cost_usd=3.0)
    assert ledger.update_outcome("a", unmeasured, True, baseline_cost_usd=4.0)
    report = ledger.report("a")
    assert ledger.get("a", unmeasured)["baseline_source"] == "caller"
    assert report["comparable_verified_requests"] == 1 and report["caller_baseline_savings_usd"] == 3.0
    assert "caller" in report["caller_baseline_scope"] and "caller-supplied" in report["savings_scope"]
    assert ledger.update_outcome("a", unmeasured, True, baseline_cost_usd=None)
    assert ledger.get("a", unmeasured)["baseline_source"] is None
    with pytest.raises(ValueError):
        row(ledger, baseline_cost_usd=1.0, baseline_source="customer")
    with pytest.raises(ValueError):
        row(ledger, baseline_source="caller")  # a source of no baseline
    with pytest.raises(ValueError):
        row(ledger, baseline_cost_usd=1.0, baseline_estimated=True, baseline_source="proxy_measured")


def test_a_caller_cannot_replace_a_cost_the_proxy_recorded_on_a_measured_request():
    """Verified savings rest on the measured baseline and on the costs beside it, so a
    caller who could rewrite those costs could set the verified figure at will."""
    from tine.ledger import MeasuredCostError
    ledger = Ledger()
    measured = row(ledger, cost_usd=1.0, routing_cost_usd=0.0, baseline_cost_usd=1.0)
    assert ledger.update_outcome("a", measured, True)
    for fields in ({"cost_usd": 0.0}, {"cost_usd": 50.0}, {"routing_cost_usd": 0.5}, {"cost_usd": None},
                   {"routing_cost_usd": None}, {"cost_usd": 0.0, "routing_cost_usd": 0.0}):
        with pytest.raises(MeasuredCostError):
            ledger.update_outcome("a", measured, False, **fields)
    record = ledger.get("a", measured)
    assert (record["success"], record["cost_usd"], record["routing_cost_usd"]) == (True, 1.0, 0.0)
    assert len(ledger.outcome_events("a", measured)) == 1 and ledger.report("a")["verified_savings_usd"] == 0.0
    # A cost the proxy did not record may be filled, and the caller may correct its own figure.
    unknown_routing = row(ledger, cost_usd=1.0, baseline_cost_usd=2.0)
    assert ledger.update_outcome("a", unknown_routing, True, routing_cost_usd=0.5)
    assert ledger.update_outcome("a", unknown_routing, True, routing_cost_usd=0.25)
    with pytest.raises(MeasuredCostError):
        ledger.update_outcome("a", unknown_routing, True, cost_usd=0.0, routing_cost_usd=0.25)
    assert ledger.get("a", unknown_routing)["routing_cost_usd"] == 0.25
    assert ledger.report("a")["verified_savings_usd"] == 0.75
    # Without a measured baseline nothing verified rests on the costs: the caller may replace them.
    unmeasured = row(ledger, cost_usd=1.0, routing_cost_usd=0.0)
    assert ledger.update_outcome("a", unmeasured, True, cost_usd=0.5, baseline_cost_usd=2.0)
    assert ledger.report("a")["caller_baseline_savings_usd"] == 1.5
    assert "cannot be replaced" in ledger.report("a")["savings_scope"]


def test_every_outcome_update_is_kept_in_an_append_only_history():
    import sqlite3
    ledger = Ledger()
    request_id = row(ledger, cost_usd=0.5)
    assert ledger.update_outcome("a", request_id, True, cost_usd=0.25, baseline_cost_usd=1.0)
    assert ledger.update_outcome("a", request_id, False, cost_usd=0.75)
    assert not ledger.update_outcome("b", request_id, True)
    events = ledger.outcome_events("a", request_id)
    assert [event["fields"] for event in events] == [
        {"success": True, "cost_usd": 0.25, "baseline_cost_usd": 1.0, "baseline_source": "caller"},
        {"success": False, "cost_usd": 0.75}]
    # What each update replaced, so the proxy's own figures are not lost.
    assert [event["previous"] for event in events] == [
        {"success": None, "cost_usd": 0.5, "baseline_cost_usd": None, "baseline_source": None},
        {"success": True, "cost_usd": 0.25}]
    assert events[0]["outcome_sequence"] < events[1]["outcome_sequence"] == ledger.get("a", request_id)["outcome_sequence"]
    assert all(event["request_id"] == request_id and event["created_at"] for event in events)
    assert ledger.outcome_events("b", request_id) == []
    for statement in ("UPDATE outcome_events SET fields_json='{}'", "DELETE FROM outcome_events"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            with ledger._db:
                ledger._db.execute(statement)
    assert len(ledger.outcome_events("a", request_id)) == 2


def test_a_routed_row_names_the_policy_digest_and_the_stage():
    ledger = Ledger()
    digest = "ab" * 32
    served = row(ledger, model="cheap", policy_sha256=digest, route_stage=1)
    escalated = row(ledger, policy_sha256=digest, route_stage="frontier")
    assert (ledger.get("a", served)["policy_sha256"], ledger.get("a", served)["route_stage"]) == (digest, "1")
    assert ledger.get("a", escalated)["route_stage"] == "frontier"
    assert ledger.get("a", row(ledger))["policy_sha256"] is None
    for invalid in ({"policy_sha256": "AB" * 32}, {"policy_sha256": "ab"}, {"policy_sha256": 1},
                    {"route_stage": -1}, {"route_stage": True}, {"route_stage": "stage-0"}, {"route_stage": 1.0}):
        with pytest.raises(ValueError):
            row(ledger, **invalid)


def test_a_ledger_from_before_baseline_sources_and_policy_traces_is_upgraded(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite3"
    ledger = Ledger(path)
    untouched = row(ledger, cost_usd=1.0, routing_cost_usd=0.0, baseline_cost_usd=4.0)
    updated = row(ledger, cost_usd=1.0, routing_cost_usd=0.0, baseline_cost_usd=3.0)
    ledger.update_outcome("a", updated, True)
    estimated = row(ledger, model="cheap", success=True, cost_usd=1.0, routing_cost_usd=0.0, baseline_cost_usd=2.0,
                    baseline_estimated=True)
    ledger.close()
    with sqlite3.connect(path) as database:
        for name in ("baseline_source", "policy_sha256", "route_stage"):
            database.execute(f"ALTER TABLE requests DROP COLUMN {name}")
        database.execute("DROP TABLE outcome_events")
    # Read without upgrading, as the report script does for a read-only file. Once an
    # outcome was reported the caller may have replaced the baseline: it is not verified.
    old = Ledger.__new__(Ledger)
    old._lock, old._db = __import__("threading").RLock(), sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    old._db.row_factory = sqlite3.Row
    before = old.report("a")
    old.close()
    assert (before["comparable_verified_requests"], before["caller_baseline_requests"],
            before["estimated_baseline_requests"]) == (0, 1, 1)
    upgraded = Ledger(path)
    assert [upgraded.get("a", request_id)["baseline_source"] for request_id in (untouched, updated, estimated)] == [
        "proxy_measured", "caller", "estimated"]
    assert upgraded.get("a", untouched)["policy_sha256"] is upgraded.get("a", untouched)["route_stage"] is None
    assert upgraded.report("a") == before
    assert upgraded.update_outcome("a", updated, False) and len(upgraded.outcome_events("a", updated)) == 1
    upgraded.close()


def test_holdout_estimate_shows_no_figure_below_the_minimum_count():
    estimate = holdout_estimate([0.5] * HOLDOUT_MIN_REQUESTS, [1.0] * (HOLDOUT_MIN_REQUESTS - 1),
                                routed_unknown=[0.2, 0.0])
    assert estimate["routed_requests"] == HOLDOUT_MIN_REQUESTS and estimate["holdout_requests"] == HOLDOUT_MIN_REQUESTS - 1
    assert estimate["routed_unknown_cost_requests"] == 2 and estimate["saving_is_upper_limit"]
    for name in ("routed_mean_cost_usd", "holdout_mean_cost_usd", "saving_fraction", "saving_fraction_low",
                 "saving_fraction_high"):
        assert estimate[name] is None
    # A held-out group that cost nothing gives no ratio either.
    assert holdout_estimate([0.5] * 40, [0.0] * 40)["saving_fraction"] is None


def test_holdout_estimate_is_one_minus_the_ratio_of_means_with_a_delta_method_interval():
    import statistics

    routed = [0.1, 0.3] * 30 + [1.0] * 10      # mean 0.3, escalations included
    holdout = [0.8, 1.2] * 20                   # mean 1.0
    estimate = holdout_estimate(routed, holdout)
    ratio = statistics.fmean(routed) / statistics.fmean(holdout)
    error = (statistics.variance(routed) / len(routed) + ratio**2 * statistics.variance(holdout) / len(holdout)) ** 0.5
    assert estimate["routed_mean_cost_usd"] == pytest.approx(statistics.fmean(routed))
    assert estimate["holdout_mean_cost_usd"] == pytest.approx(1.0)
    assert estimate["saving_fraction"] == pytest.approx(1 - ratio)
    assert estimate["saving_fraction_low"] == pytest.approx(1 - ratio - 1.959964 * error, abs=1e-6)
    assert estimate["saving_fraction_high"] == pytest.approx(1 - ratio + 1.959964 * error, abs=1e-6)
    assert estimate["confidence"] == 0.95 and "delta method" in estimate["method"]
    for words in ("estimate from a random sample", "assumes the two groups are alike", "does not measure quality"):
        assert words in estimate["scope"]
    # Routing that costs more than the frontier is a negative saving, not hidden.
    assert holdout_estimate([2.0, 2.2] * 20, [1.0, 1.2] * 20)["saving_fraction"] < 0


def test_report_compares_the_routed_and_held_out_groups_and_counts_unknown_costs():
    ledger = Ledger()
    assert ledger.report("a")["holdout_estimate"] is None
    for _ in range(HOLDOUT_MIN_REQUESTS):
        row(ledger, model="small", cost_usd=0.1, routing_cost_usd=0.1, holdout=False)
        row(ledger, cost_usd=1.0, routing_cost_usd=0.0, baseline_cost_usd=1.0, holdout=True)
    row(ledger, cost_usd=5.0, routing_cost_usd=0.0)                    # not in the comparison
    row(ledger, tenant="b", cost_usd=9.0, routing_cost_usd=0.0, holdout=False)
    estimate = ledger.report("a")["holdout_estimate"]
    assert estimate["routed_requests"] == HOLDOUT_MIN_REQUESTS == estimate["holdout_requests"]
    assert estimate["routed_unknown_cost_requests"] == 0 == estimate["holdout_unknown_cost_requests"]
    assert estimate["saving_is_upper_limit"] is False
    assert estimate["routed_mean_cost_usd"] == pytest.approx(0.2)
    assert estimate["saving_fraction"] == pytest.approx(0.8)
    assert estimate["saving_fraction_low"] == pytest.approx(0.8) == estimate["saving_fraction_high"]
    row(ledger, cost_usd=1.0, routing_cost_usd=None, holdout=True)     # a held-out row with no price: left out
    assert ledger.report("a")["holdout_estimate"]["holdout_unknown_cost_requests"] == 1
    with pytest.raises(ValueError, match="holdout"):
        row(ledger, holdout=1)


def test_routed_requests_whose_cost_is_not_fully_known_count_at_their_known_part():
    # Escalations after a failed cheap call, a check that raised or the route deadline
    # have routing cost unknown but paid the frontier. Leaving them out would keep only
    # the cheap routed requests and overstate the saving (0.988 here instead of <= 0.494).
    ledger = Ledger()
    for _ in range(40):
        row(ledger, cost_usd=1.0, routing_cost_usd=0.0, baseline_cost_usd=1.0, holdout=True)
        row(ledger, model="small", cost_usd=0.01, routing_cost_usd=0.002, holdout=False)
        row(ledger, cost_usd=1.0, routing_cost_usd=None, holdout=False)           # escalated
    row(ledger, model="small", cost_usd=None, routing_cost_usd=0.002, holdout=False)  # no usage: counted at 0.002
    estimate = ledger.report("a")["holdout_estimate"]
    assert estimate["routed_requests"] == 40 and estimate["routed_unknown_cost_requests"] == 41
    assert estimate["saving_is_upper_limit"] is True
    known = [0.012] * 40 + [1.0] * 40 + [0.002]
    assert estimate["routed_mean_cost_usd"] == pytest.approx(sum(known) / len(known))
    assert estimate["saving_fraction"] == pytest.approx(1 - sum(known) / len(known))
    assert estimate["saving_fraction"] < 0.51 and estimate["saving_fraction_high"] < 0.62
    assert "upper limit" in estimate["scope"]
    # Partial routed costs alone are enough to start the comparison.
    alone = Ledger()
    row(alone, cost_usd=None, routing_cost_usd=None, holdout=False)
    assert alone.report("a")["holdout_estimate"]["routed_unknown_cost_requests"] == 1


def test_costs_of_a_request_in_the_holdout_comparison_cannot_be_set_by_the_caller():
    ledger = Ledger()
    routed = row(ledger, model="small", cost_usd=0.1, routing_cost_usd=None, holdout=False)
    held = row(ledger, cost_usd=1.0, routing_cost_usd=0.0, baseline_cost_usd=1.0, holdout=True)
    for request_id in (routed, held):
        for name in ("cost_usd", "routing_cost_usd"):
            with pytest.raises(HoldoutCostError):
                ledger.update_outcome("a", request_id, True, **{name: 0.0})
        assert ledger.outcome_events("a", request_id) == []
    # The outcome itself, and a baseline the proxy did not measure, can still be set.
    assert ledger.update_outcome("a", routed, True, baseline_cost_usd=1.0)
    assert ledger.get("a", routed)["success"] is True and ledger.get("a", routed)["routing_cost_usd"] is None


def test_holdout_column_is_added_to_an_older_ledger(tmp_path):
    import sqlite3

    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    request_id = row(ledger, cost_usd=0.25, routing_cost_usd=0.0)
    ledger.close()
    with sqlite3.connect(path) as database:
        database.execute("ALTER TABLE requests DROP COLUMN holdout")
    migrated = Ledger(path)
    assert migrated.get("a", request_id)["holdout"] is None
    assert migrated.report("a")["holdout_estimate"] is None
    assert migrated.get("a", row(migrated, holdout=True))["holdout"] == 1
    migrated.close()


def test_the_checks_column_is_added_to_an_older_ledger_and_keeps_check_records(tmp_path):
    import sqlite3

    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    request_id = row(ledger, cost_usd=0.25)
    ledger.close()
    with sqlite3.connect(path) as database:
        database.execute("ALTER TABLE requests DROP COLUMN checks_json")
    migrated = Ledger(path)
    # An older row does not say what checks ran: unknown, not an empty list.
    assert migrated.get("a", request_id)["checks"] is None
    checks = [{"check": "tests", "model": "cheap", "checked_model": "cheap", "passed": False, "cost_usd": 0.01,
               "error": None, "steps": [{"type": "webhook", "passed": False, "cost_usd": 0.01}]}]
    checked = row(migrated, checks=checks)
    assert migrated.get("a", checked)["checks"] == checks
    assert migrated.get("a", row(migrated, checks=[]))["checks"] == []
    for broken in ({"check": "tests"}, ["not a record"]):
        with pytest.raises(ValueError, match="checks"):
            row(migrated, checks=broken)
    migrated.close()


def test_a_backup_under_an_open_writer_holds_every_committed_row_in_one_file(tmp_path):
    import shutil
    import sqlite3

    path = tmp_path / "live" / "ledger.sqlite3"
    ledger = Ledger(path)  # stays open, as the proxy's does
    ids = sorted(row(ledger, cost_usd=0.25) for _ in range(3))
    # What an operator hits: the rows are still in the WAL, so the main file alone has no tables.
    assert path.with_name("ledger.sqlite3-wal").stat().st_size > 0
    naive = tmp_path / "naive" / "ledger.sqlite3"
    naive.parent.mkdir()
    shutil.copy(path, naive)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        sqlite3.connect(naive).execute("SELECT COUNT(*) FROM requests")

    writer = sqlite3.connect(path, isolation_level=None)  # a write in flight, not committed
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("UPDATE requests SET model='half-written'")
    output = tmp_path / "out" / "copy.sqlite3"
    result = backup_ledger(path, output)
    assert result == {"ledger": str(path), "output": str(output), "bytes": output.stat().st_size,
                      "rows": {"requests": 3, "outcome_events": 0}}
    writer.execute("COMMIT")
    writer.close()
    late = row(ledger)  # the backup neither blocked the ledger nor changed it
    assert ledger.get("a", late)["model"] == "frontier" and ledger.get("a", ids[0])["model"] == "half-written"

    # One file, private, with a rollback journal: readable where nothing can be written.
    assert [item.name for item in output.parent.iterdir()] == ["copy.sqlite3"]
    assert output.stat().st_mode & 0o777 == 0o600 and output.read_bytes()[18:20] == b"\x01\x01"
    output.parent.chmod(0o500)
    try:
        copy = sqlite3.connect(f"{output.as_uri()}?mode=ro", uri=True)
        rows = copy.execute("SELECT request_id, model, cost_usd FROM requests ORDER BY request_id").fetchall()
        copy.close()
        assert [item.name for item in output.parent.iterdir()] == ["copy.sqlite3"]
    finally:
        output.parent.chmod(0o700)
    assert rows == [(request_id, "frontier", 0.25) for request_id in ids]  # the committed snapshot only
    ledger.close()


def test_a_backup_never_replaces_a_file_unasked_or_the_ledger_itself(tmp_path):
    import sqlite3

    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    request_id = row(ledger)
    output = tmp_path / "out" / "copy.sqlite3"
    output.parent.mkdir()
    output.write_text("an earlier backup", encoding="utf-8")
    output.chmod(0o644)
    with pytest.raises(FileExistsError, match="--force"):
        backup_ledger(path, output)
    assert output.read_text(encoding="utf-8") == "an earlier backup"
    assert backup_ledger(path, output, force=True)["rows"]["requests"] == 1
    assert output.stat().st_mode & 0o777 == 0o600
    for force in (False, True):
        with pytest.raises(ValueError, match="the ledger itself"):
            backup_ledger(path, path, force=force)
    assert ledger.get("a", request_id) is not None

    # A source that is not a ledger leaves nothing behind, and a missing one is not created.
    other = tmp_path / "other.sqlite3"
    with sqlite3.connect(other) as database:
        database.execute("CREATE TABLE notes (text TEXT)")
    text = tmp_path / "notes.txt"
    text.write_text("not a database, and long enough to have a header to read. " * 4, encoding="utf-8")
    for source, error in ((other, ValueError), (text, sqlite3.DatabaseError),
                          (tmp_path / "missing.sqlite3", FileNotFoundError)):
        with pytest.raises(error):
            backup_ledger(source, tmp_path / "out" / "second.sqlite3")
    assert not (tmp_path / "missing.sqlite3").exists()
    with pytest.raises(IsADirectoryError):
        backup_ledger(path, tmp_path / "out")
    with pytest.raises(OSError, match="Cannot write in .*notes.txt .*put the copy where files can be written"):
        backup_ledger(path, text / "copy.sqlite3")  # as on the container's read-only root
    assert [item.name for item in output.parent.iterdir()] == ["copy.sqlite3"]
    ledger.close()


def test_backup_ledger_command_reads_the_ledger_the_proxy_does(tmp_path, monkeypatch, capsys):
    import json

    from tine.__main__ import main
    from tine.proxy import ProxyConfig

    path, other = tmp_path / "ledger.sqlite3", tmp_path / "other.sqlite3"
    ledger = Ledger(path)
    row(ledger)
    Ledger(other).close()
    output = tmp_path / "copy.sqlite3"
    monkeypatch.setenv("TINE_LEDGER_PATH", str(path))
    main(["backup-ledger", "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["rows"]["requests"] == 1
    with pytest.raises(SystemExit) as stopped:
        main(["backup-ledger", "--output", str(output)])
    assert "tine backup-ledger:" in str(stopped.value) and "already exists" in str(stopped.value)
    closed = other.read_bytes()
    main(["backup-ledger", "--ledger", str(other), "--output", str(output), "--force"])
    assert json.loads(capsys.readouterr().out)["rows"]["requests"] == 0
    # A ledger nobody has open is not changed; only an empty -wal and a -shm appear beside it.
    assert other.read_bytes() == closed and other.with_name("other.sqlite3-wal").stat().st_size == 0
    # With neither given it looks where the proxy would, and names the path it tried.
    monkeypatch.delenv("TINE_LEDGER_PATH")
    monkeypatch.chdir(tmp_path)
    assert ProxyConfig.__dataclass_fields__["ledger_path"].default == DEFAULT_LEDGER_PATH
    with pytest.raises(SystemExit) as stopped:
        main(["backup-ledger", "--output", str(tmp_path / "none.sqlite3")])
    assert f"No ledger file at {DEFAULT_LEDGER_PATH}" in str(stopped.value)
    ledger.close()


def test_backup_ledger_command_writes_only_beside_its_output(tmp_path):
    """As in the container: the working directory cannot be written, there is no home or
    temporary directory, and another process has the ledger open."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    path = tmp_path / "data" / "ledger.sqlite3"
    ledger = Ledger(path)
    row(ledger)
    root = tmp_path / "root"
    root.mkdir(mode=0o500)
    environment = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(Path(__file__).parents[1]),
                   "PYTHONDONTWRITEBYTECODE": "1", "TINE_LEDGER_PATH": str(path),
                   "HOME": "/nonexistent", "TMPDIR": "/nonexistent", "SQLITE_TMPDIR": "/nonexistent"}
    done = subprocess.run(
        [sys.executable, "-m", "tine", "backup-ledger", "--output", str(path.with_name("copy.sqlite3"))],
        cwd=root, env=environment, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert '"requests": 1' in done.stdout and list(root.iterdir()) == []
    assert sorted(item.name for item in path.parent.iterdir()) == [
        "copy.sqlite3", "ledger.sqlite3", "ledger.sqlite3-shm", "ledger.sqlite3-wal"]
    ledger.close()
