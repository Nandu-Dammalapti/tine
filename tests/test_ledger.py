import pytest
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

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
    assert "always sent to the frontier, which is not built" in report["savings_scope"]


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
    ledger.update_outcome("a", request_id, True, cost_usd=0.5)
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
    assert ledger.update_outcome("a", measured, True, cost_usd=0.5)
    assert ledger.report("a")["verified_savings_usd"] == 0.5
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
