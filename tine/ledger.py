"""Tenant-scoped SQLite development ledger.

Costs and outcomes are deliberately nullable: a successful HTTP response is not
evidence that a task succeeded, and missing usage/pricing is not a zero cost.
This is a local development store, not the planned production Postgres backend.

Every baseline carries where it came from (``baseline_source``): measured by the
proxy, supplied by the caller, or estimated. Only measured ones count as verified
savings. ``requests`` holds the current state of a request; every outcome update is
also appended to ``outcome_events`` with the values it replaced, so none is lost.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_UNSET = object()
MAX_TOKEN_COUNT = 2**63 - 1  # SQLite INTEGER storage limit.
# Where a baseline cost came from. Only a baseline the proxy measured, by calling the
# frontier model for this request, counts toward verified savings.
BASELINE_SOURCES = ("proxy_measured", "caller", "estimated")


class MeasuredBaselineError(ValueError):
    """An outcome update tried to replace a baseline the proxy measured itself."""


class MeasuredCostError(ValueError):
    """An outcome update tried to replace a cost the proxy recorded on a request whose
    baseline it measured: verified savings rest on both."""


def _legacy_baseline_source(columns) -> str:
    """SQL for where a row's baseline came from, in a ledger written before the column.

    Such a ledger cannot say. A baseline on a row whose outcome was never updated was
    written by the proxy; once an outcome arrived, the caller may have replaced it, so
    it is counted as the caller's. That can only move savings out of the verified total.
    """
    estimated = "baseline_estimated" if "baseline_estimated" in columns else "0"
    untouched = "outcome_updated_at IS NULL" if "outcome_updated_at" in columns else "success IS NULL"
    return (f"CASE WHEN baseline_cost_usd IS NULL THEN NULL WHEN {estimated} THEN 'estimated' "
            f"WHEN {untouched} THEN 'proxy_measured' ELSE 'caller' END")


def request_fingerprint(
    provider: str, body: dict[str, Any], *, cache_context: dict[str, Any] | None = None,
) -> str:
    """Hash request semantics and optional upstream context, scoped by tenant in SQL."""
    payload = {"version": 1, "provider": provider, "body": body}
    if cache_context is not None:
        payload.update(version=2, cache_context=cache_context)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonnegative(value: Any, name: str) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative or None")


def _cost_sum(values) -> float | None:
    """Unrepresentable totals are unknown, never Infinity in the API response."""
    try:
        total = math.fsum(values)
    except OverflowError:
        return None
    return total if math.isfinite(total) else None


class Ledger:
    """A single local DB can contain tenants; every read/update requires one."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            db_path = Path(path)
            db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not db_path.exists():
                db_path.touch(mode=0o600)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute("PRAGMA journal_mode=WAL")
            # Serialize schema upgrades across processes, including column checks.
            self._db.execute("BEGIN IMMEDIATE")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    tenant TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    requested_model TEXT,
                    created_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    fingerprint TEXT NOT NULL,
                    upstream_status INTEGER,
                    stream INTEGER NOT NULL DEFAULT 0,
                    fallback_used INTEGER NOT NULL DEFAULT 0,
                    cached_from TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cached_tokens INTEGER,
                    cost_usd REAL,
                    routing_cost_usd REAL,
                    baseline_cost_usd REAL,
                    latency_ms REAL,
                    outcome_updated_at TEXT,
                    outcome_sequence INTEGER NOT NULL DEFAULT 0,
                    success INTEGER CHECK(success IN (0, 1) OR success IS NULL),
                    baseline_estimated INTEGER NOT NULL DEFAULT 0,
                    route_skip_reason TEXT,
                    baseline_source TEXT,
                    policy_sha256 TEXT,
                    route_stage TEXT
                )"""
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS verified_cache "
                "ON requests(tenant, fingerprint, success)"
            )
            columns = {
                row["name"] for row in self._db.execute("PRAGMA table_info(requests)")
            }
            if "outcome_updated_at" not in columns:
                self._db.execute("ALTER TABLE requests ADD COLUMN outcome_updated_at TEXT")
                self._db.execute(
                    "UPDATE requests SET outcome_updated_at=created_at "
                    "WHERE success IS NOT NULL"
                )
            if "outcome_sequence" not in columns:
                self._db.execute("ALTER TABLE requests ADD COLUMN outcome_sequence INTEGER NOT NULL DEFAULT 0")
            if "routing_cost_usd" not in columns:
                # Older records never measured routing overhead. Do not invent zero.
                self._db.execute("ALTER TABLE requests ADD COLUMN routing_cost_usd REAL")
            if "baseline_estimated" not in columns:
                # Older records hold baselines that were measured or supplied by the caller.
                self._db.execute("ALTER TABLE requests ADD COLUMN baseline_estimated INTEGER NOT NULL DEFAULT 0")
                columns.add("baseline_estimated")
            if "route_skip_reason" not in columns:
                self._db.execute("ALTER TABLE requests ADD COLUMN route_skip_reason TEXT")
            if "baseline_source" not in columns:
                self._db.execute("ALTER TABLE requests ADD COLUMN baseline_source TEXT")
                self._db.execute(f"UPDATE requests SET baseline_source={_legacy_baseline_source(columns)}")
            for name in ("policy_sha256", "route_stage"):
                if name not in columns:
                    # Older rows do not say which policy, if any, served them.
                    self._db.execute(f"ALTER TABLE requests ADD COLUMN {name} TEXT")
            # Append-only history of outcome updates: what each one set, and what it replaced.
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS outcome_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    outcome_sequence INTEGER NOT NULL,
                    fields_json TEXT NOT NULL,
                    previous_json TEXT NOT NULL
                )"""
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS outcome_event_request ON outcome_events(tenant, request_id, event_id)"
            )
            for action in ("UPDATE", "DELETE"):
                self._db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS outcome_events_no_{action.lower()} BEFORE {action} "
                    "ON outcome_events BEGIN SELECT RAISE(ABORT, 'outcome_events is append-only'); END"
                )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS ledger_state "
                "(id INTEGER PRIMARY KEY CHECK(id=1), outcome_sequence INTEGER NOT NULL)"
            )
            self._db.execute("INSERT OR IGNORE INTO ledger_state VALUES(1, 0)")
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS cache_outcome_order ON requests"
                "(tenant, fingerprint, outcome_sequence DESC, outcome_updated_at DESC, created_at DESC)"
            )

    def _next_outcome_sequence(self) -> int:
        """Allocate under the same write transaction as its associated outcome."""
        self._db.execute("UPDATE ledger_state SET outcome_sequence=outcome_sequence+1 WHERE id=1")
        return self._db.execute("SELECT outcome_sequence FROM ledger_state WHERE id=1").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def record(
        self,
        *,
        tenant: str,
        provider: str,
        model: str,
        request_body: dict[str, Any],
        response_body: dict[str, Any] | None = None,
        requested_model: str | None = None,
        request_id: str | None = None,
        success: bool | None = None,
        upstream_status: int | None = None,
        stream: bool = False,
        fallback_used: bool = False,
        cached_from: str | None = None,
        cache_context: dict[str, Any] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
        routing_cost_usd: float | None = None,
        baseline_cost_usd: float | None = None,
        latency_ms: float | None = None,
        baseline_estimated: bool = False,
        route_skip_reason: str | None = None,
        baseline_source: str | None = None,
        policy_sha256: str | None = None,
        route_stage: int | str | None = None,
    ) -> str:
        """Store one request. ``baseline_estimated`` marks a ``baseline_cost_usd`` that
        was worked out, not measured: the frontier model was not called for this
        request. The report keeps such rows out of verified savings.
        ``route_skip_reason`` says why a route policy's cheaper stages were not tried.

        ``baseline_source`` says where a baseline came from (``BASELINE_SOURCES``); left
        out, a baseline is ``estimated`` when so marked and ``proxy_measured`` otherwise.
        ``policy_sha256`` and ``route_stage`` trace a request a route policy ran on: the
        digest of the policy file and the index of the stage that answered, or
        ``"frontier"``. The stage is stored as text."""
        if not all(isinstance(x, str) and x for x in (tenant, provider, model)):
            raise ValueError("tenant, provider, and model must be nonempty strings")
        if not isinstance(baseline_estimated, bool) or (baseline_estimated and baseline_cost_usd is None):
            raise ValueError("baseline_estimated must be a bool and needs a baseline_cost_usd")
        if baseline_source is not None and (baseline_source not in BASELINE_SOURCES or baseline_cost_usd is None):
            raise ValueError(f"baseline_source must be one of {BASELINE_SOURCES} and needs a baseline_cost_usd")
        if baseline_cost_usd is not None:
            baseline_source = baseline_source or ("estimated" if baseline_estimated else "proxy_measured")
            if baseline_estimated and baseline_source != "estimated":
                raise ValueError("an estimated baseline must have the source 'estimated'")
        if policy_sha256 is not None and (
                not isinstance(policy_sha256, str) or len(policy_sha256) != 64
                or set(policy_sha256) - set("0123456789abcdef")):
            raise ValueError("policy_sha256 must be a lowercase hexadecimal SHA-256 digest or None")
        if route_stage is not None and route_stage != "frontier" and (
                isinstance(route_stage, bool) or not isinstance(route_stage, int) or route_stage < 0):
            raise ValueError("route_stage must be a stage index, 'frontier' or None")
        if success is not None and not isinstance(success, bool):
            raise ValueError("success must be bool or None")
        if route_skip_reason is not None and (not isinstance(route_skip_reason, str) or not route_skip_reason):
            raise ValueError("route_skip_reason must be a nonempty string or None")
        for name, value in (
            ("cost_usd", cost_usd),
            ("routing_cost_usd", routing_cost_usd),
            ("baseline_cost_usd", baseline_cost_usd),
            ("latency_ms", latency_ms),
        ):
            _nonnegative(value, name)
        for name, value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
            ("cached_tokens", cached_tokens),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_TOKEN_COUNT
            ):
                raise ValueError(f"{name} must be an integer between 0 and {MAX_TOKEN_COUNT} or None")
        request_id = request_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        values = {
            "request_id": request_id,
            "tenant": tenant,
            "provider": provider,
            "model": model,
            "requested_model": requested_model,
            "created_at": now,
            "request_json": json.dumps(request_body, allow_nan=False),
            "response_json": (
                json.dumps(response_body, allow_nan=False)
                if response_body is not None
                else None
            ),
            "fingerprint": request_fingerprint(provider, request_body, cache_context=cache_context),
            "upstream_status": upstream_status,
            "stream": int(stream),
            "fallback_used": int(fallback_used),
            "cached_from": cached_from,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost_usd,
            "routing_cost_usd": routing_cost_usd,
            "baseline_cost_usd": baseline_cost_usd,
            "latency_ms": latency_ms,
            "outcome_updated_at": now if success is not None else None,
            "outcome_sequence": 0,
            "success": None if success is None else int(success),
            "baseline_estimated": int(baseline_source == "estimated"),
            "route_skip_reason": route_skip_reason,
            "baseline_source": baseline_source,
            "policy_sha256": policy_sha256,
            "route_stage": None if route_stage is None else str(route_stage),
        }
        columns = ", ".join(values)
        placeholders = ", ".join(f":{key}" for key in values)
        with self._lock, self._db:
            if success is not None:
                values["outcome_sequence"] = self._next_outcome_sequence()
            self._db.execute(
                f"INSERT INTO requests ({columns}) VALUES ({placeholders})", values
            )
        return request_id

    def get(self, tenant: str, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM requests WHERE tenant=? AND request_id=?",
                (tenant, request_id),
            ).fetchone()
        return self._decode(row) if row is not None else None

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["request_body"] = json.loads(result.pop("request_json"))
        raw_response = result.pop("response_json")
        result["response_body"] = json.loads(raw_response) if raw_response else None
        result["success"] = (
            None if result["success"] is None else bool(result["success"])
        )
        return result

    def update_outcome(
        self,
        tenant: str,
        request_id: str,
        success: bool,
        *,
        cost_usd: Any = _UNSET,
        routing_cost_usd: Any = _UNSET,
        baseline_cost_usd: Any = _UNSET,
    ) -> bool:
        """Apply a caller-supplied outcome and optional costs later.

        A baseline supplied here is the caller's (``baseline_source`` ``caller``). It may
        replace an estimated or caller-supplied one, never one the proxy measured: that
        raises ``MeasuredBaselineError`` and changes nothing. On a request whose baseline
        the proxy measured, ``cost_usd`` and ``routing_cost_usd`` may fill a cost that is
        not known or replace one an earlier update supplied, never one the proxy recorded:
        that raises ``MeasuredCostError`` and changes nothing. Each update is appended to
        ``outcome_events`` with the values it replaced. Returns False when the tenant
        has no such request."""
        if not isinstance(success, bool):
            raise ValueError("success must be a bool")
        fields: dict[str, Any] = {"success": success}
        for name, value in (
            ("cost_usd", cost_usd),
            ("routing_cost_usd", routing_cost_usd),
            ("baseline_cost_usd", baseline_cost_usd),
        ):
            if value is not _UNSET:
                _nonnegative(value, name)
                fields[name] = value
        if "baseline_cost_usd" in fields:
            fields["baseline_source"] = None if fields["baseline_cost_usd"] is None else "caller"
        now = datetime.now(timezone.utc).isoformat()
        changes = {**fields, "success": int(success), "outcome_updated_at": now}
        if "baseline_cost_usd" in fields:
            changes["baseline_estimated"] = 0
        with self._lock, self._db:
            # Hold the write lock from the read of the old values to the event that keeps them.
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT success, cost_usd, routing_cost_usd, baseline_cost_usd, baseline_source "
                "FROM requests WHERE tenant=? AND request_id=?",
                (tenant, request_id),
            ).fetchone()
            if row is None:
                return False
            if "baseline_cost_usd" in fields and row["baseline_source"] == "proxy_measured":
                raise MeasuredBaselineError("the proxy measured this request's baseline; it cannot be replaced")
            recorded = [name for name in ("cost_usd", "routing_cost_usd") if name in fields and row[name] is not None]
            if recorded and row["baseline_source"] == "proxy_measured":
                # A set cost is the proxy's unless an earlier outcome update supplied it.
                supplied = {name for event in self._db.execute(
                    "SELECT fields_json FROM outcome_events WHERE tenant=? AND request_id=?", (tenant, request_id))
                    for name in json.loads(event["fields_json"])}
                if any(name not in supplied for name in recorded):
                    raise MeasuredCostError("the proxy recorded this request's costs and measured its baseline; "
                                            "a recorded cost cannot be replaced")
            previous = {name: row[name] for name in fields}
            previous["success"] = None if row["success"] is None else bool(row["success"])
            changes["outcome_sequence"] = sequence = self._next_outcome_sequence()
            self._db.execute(
                f"UPDATE requests SET {', '.join(f'{name}=?' for name in changes)} "
                "WHERE tenant=? AND request_id=?",
                [*changes.values(), tenant, request_id],
            )
            self._db.execute(
                "INSERT INTO outcome_events (tenant, request_id, created_at, outcome_sequence, fields_json, "
                "previous_json) VALUES (?, ?, ?, ?, ?, ?)",
                (tenant, request_id, now, sequence, json.dumps(fields, allow_nan=False),
                 json.dumps(previous, allow_nan=False)),
            )
        return True

    def fill_usage(
        self,
        tenant: str,
        request_id: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> bool:
        """Fill token counts and the answer cost the proxy learned after recording a
        request, such as a streamed reply's usage once the stream has ended. Only empty
        fields are filled: a value already there, such as a cost the caller supplied
        meanwhile, is kept. This is not an outcome update and adds no outcome event.
        Returns False when the tenant has no such request."""
        _nonnegative(cost_usd, "cost_usd")
        for name, value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
            ("cached_tokens", cached_tokens),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_TOKEN_COUNT
            ):
                raise ValueError(f"{name} must be an integer between 0 and {MAX_TOKEN_COUNT} or None")
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE requests SET input_tokens=COALESCE(input_tokens, ?), output_tokens=COALESCE(output_tokens, ?), "
                "cached_tokens=COALESCE(cached_tokens, ?), cost_usd=COALESCE(cost_usd, ?) "
                "WHERE tenant=? AND request_id=?",
                (input_tokens, output_tokens, cached_tokens, cost_usd, tenant, request_id),
            )
        return cursor.rowcount > 0

    def outcome_events(self, tenant: str, request_id: str) -> list[dict[str, Any]]:
        """Every outcome update of one request, oldest first: the fields it set and the
        values they held before. Updates made before the table existed are not here."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM outcome_events WHERE tenant=? AND request_id=? ORDER BY event_id",
                (tenant, request_id),
            ).fetchall()
        return [
            {**{key: row[key] for key in ("event_id", "request_id", "created_at", "outcome_sequence")},
             "fields": json.loads(row["fields_json"]), "previous": json.loads(row["previous_json"])}
            for row in rows
        ]

    def cached(
        self, tenant: str, provider: str, body: dict, *, cache_context: dict[str, Any] | None = None,
    ) -> dict | None:
        """Use the latest resolved outcome; later failures invalidate old success.

        An unresolved newer request is not evidence that a previously verified
        answer failed. A database sequence orders resolved outcomes independently
        of wall-clock changes. Legacy entries retain timestamp ordering until
        newer sequenced evidence arrives.
        Context-aware lookups require a matching context-aware fingerprint.
        """
        with self._lock:
            row = self._db.execute(
                """SELECT * FROM requests
                   WHERE tenant=? AND fingerprint=? AND success IS NOT NULL
                   ORDER BY outcome_sequence DESC, outcome_updated_at DESC,
                            created_at DESC, rowid DESC LIMIT 1""",
                (tenant, request_fingerprint(provider, body, cache_context=cache_context)),
            ).fetchone()
        if (
            row is None
            or row["success"] != 1
            or row["upstream_status"] is None
            or not 200 <= row["upstream_status"] < 300
            or row["stream"]
            or row["response_json"] is None
        ):
            return None
        return self._decode(row)

    def report(self, tenant: str) -> dict[str, Any]:
        """Report known ledger amounts, never extrapolate missing costs.

        Verified savings rest on baselines the proxy measured only. Successful requests
        whose baseline is an estimate, or was supplied by the caller, are counted and
        summed apart, under names that say so.
        """
        with self._lock:
            # A ledger opened read-only may predate the column: work its sources out as an upgrade would.
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(requests)")}
            source = "baseline_source" if "baseline_source" in columns else _legacy_baseline_source(columns)
            rows = self._db.execute(
                f"SELECT success,cost_usd,routing_cost_usd,baseline_cost_usd,cached_from,{source} AS baseline_source "
                "FROM requests WHERE tenant=?",
                (tenant,),
            ).fetchall()
        costed = [
            row
            for row in rows
            if row["success"] == 1
            and row["cost_usd"] is not None
            and row["routing_cost_usd"] is not None
            and row["baseline_cost_usd"] is not None
        ]
        comparable = [row for row in costed if row["baseline_source"] == "proxy_measured"]
        cost_columns = ("cost_usd", "routing_cost_usd")

        def apart(name: str) -> tuple[int, float | None]:
            """How many costed successes have a baseline from this source, and their net saving."""
            chosen = [row for row in costed if row["baseline_source"] == name]
            baseline = _cost_sum(row["baseline_cost_usd"] for row in chosen)
            spent = _cost_sum(row[column] for row in chosen for column in cost_columns)
            return len(chosen), (baseline - spent if chosen and baseline is not None and spent is not None else None)

        estimated_requests, estimated_savings = apart("estimated")
        caller_requests, caller_savings = apart("caller")
        verified_answer_cost = _cost_sum(row["cost_usd"] for row in comparable)
        verified_routing_cost = _cost_sum(row["routing_cost_usd"] for row in comparable)
        verified_cost = _cost_sum(row[column] for row in comparable for column in cost_columns)
        verified_baseline = _cost_sum(row["baseline_cost_usd"] for row in comparable)
        known_answer_cost = _cost_sum(
            row["cost_usd"] for row in rows if row["cost_usd"] is not None
        )
        known_routing_cost = _cost_sum(
            row["routing_cost_usd"] for row in rows if row["routing_cost_usd"] is not None
        )
        known_cost = _cost_sum(row[column] for row in rows for column in cost_columns if row[column] is not None)
        unknown_answer = sum(row["cost_usd"] is None for row in rows)
        unknown_routing = sum(row["routing_cost_usd"] is None for row in rows)
        unknown_cost = sum(row["cost_usd"] is None or row["routing_cost_usd"] is None for row in rows)
        successes = sum(row["success"] == 1 for row in rows)
        savings = fraction = None
        if comparable and verified_baseline is not None and verified_cost is not None:
            savings = verified_baseline - verified_cost
            if verified_baseline > 0:
                value = 1 - verified_cost / verified_baseline
                fraction = value if math.isfinite(value) else None
        return {
            "tenant": tenant,
            "requests": len(rows),
            "verified_successes": successes,
            "failures": sum(row["success"] == 0 for row in rows),
            "pending_outcomes": sum(row["success"] is None for row in rows),
            "cache_hits": sum(row["cached_from"] is not None for row in rows),
            "known_answer_cost_total_usd": known_answer_cost,
            "known_routing_cost_total_usd": known_routing_cost,
            "known_cost_total_usd": known_cost,
            "known_cost_per_verified_success_usd": (
                known_cost / successes if successes and not unknown_cost and known_cost is not None else None
            ),
            "unknown_cost_requests": unknown_cost,
            "unknown_answer_cost_requests": unknown_answer,
            "unknown_routing_cost_requests": unknown_routing,
            "comparable_verified_requests": len(comparable),
            "verified_answer_cost_usd": verified_answer_cost,
            "verified_routing_cost_usd": verified_routing_cost,
            "verified_cost_usd": verified_cost,
            "verified_baseline_cost_usd": verified_baseline,
            "verified_savings_usd": savings,
            "savings_fraction": fraction,
            "estimated_baseline_requests": estimated_requests,
            "estimated_baseline_savings_usd": estimated_savings,
            "estimated_baseline_scope": (
                "successful requests answered by a cheaper model, where the frontier model was not called: the "
                "baseline is an estimate (the returned answer's token counts at the frontier's prices), so this "
                "figure is approximate and is not part of verified savings"
            ),
            "caller_baseline_requests": caller_requests,
            "caller_baseline_savings_usd": caller_savings,
            "caller_baseline_scope": (
                "successful requests whose baseline cost was supplied by the caller through the outcome endpoint "
                "(in a ledger written before baseline sources were recorded, any request whose outcome was "
                "updated): Tine did not measure it, so this figure is the caller's and is not part of verified savings"
            ),
            "scope": (
                "development ledger; outcomes are caller-attested, verified baselines were measured by the proxy, "
                "and on those requests the caller can supply only costs the proxy did not record"
            ),
            "cost_scope": (
                "answer generation plus routing/checking costs; known totals sum only measured "
                "components and are incomplete when unknown_cost_requests is nonzero"
            ),
            "savings_scope": (
                "net of routing costs on successful requests with known answer, routing, and baseline costs only, "
                "where the proxy measured the baseline by calling the frontier model; a cost the proxy recorded on "
                "such a request cannot be replaced by the caller, and a cost it did not record (such as routing "
                "cost without a route policy) is the caller's; failed, pending, and "
                "incompletely costed requests, and requests whose baseline is estimated or caller-supplied, "
                "are excluded from this subtotal, so it is not a billing amount. Under live routing the proxy "
                "calls the frontier only for requests that escalated, which also paid for the cheaper attempts, "
                "so this subtotal is not the policy's saving; measuring that needs a random share of requests "
                "always sent to the frontier, which is not built"
            ),
        }
