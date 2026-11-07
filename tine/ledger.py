"""Tenant-scoped SQLite development ledger.

Costs and outcomes are deliberately nullable: a successful HTTP response is not
evidence that a task succeeded, and missing usage/pricing is not a zero cost.
This is a local development store, not the planned production Postgres backend.
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


def request_fingerprint(provider: str, body: dict[str, Any]) -> str:
    """Hash all semantic JSON request fields; tenant scope is enforced in SQL."""
    encoded = json.dumps(
        {"version": 1, "provider": provider, "body": body},
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
                    baseline_cost_usd REAL,
                    latency_ms REAL,
                    outcome_updated_at TEXT,
                    success INTEGER CHECK(success IN (0, 1) OR success IS NULL)
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
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
        baseline_cost_usd: float | None = None,
        latency_ms: float | None = None,
    ) -> str:
        if not all(isinstance(x, str) and x for x in (tenant, provider, model)):
            raise ValueError("tenant, provider, and model must be nonempty strings")
        if success is not None and not isinstance(success, bool):
            raise ValueError("success must be bool or None")
        for name, value in (
            ("cost_usd", cost_usd),
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
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")
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
            "fingerprint": request_fingerprint(provider, request_body),
            "upstream_status": upstream_status,
            "stream": int(stream),
            "fallback_used": int(fallback_used),
            "cached_from": cached_from,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost_usd,
            "baseline_cost_usd": baseline_cost_usd,
            "latency_ms": latency_ms,
            "outcome_updated_at": now if success is not None else None,
            "success": None if success is None else int(success),
        }
        columns = ", ".join(values)
        placeholders = ", ".join(f":{key}" for key in values)
        with self._lock, self._db:
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
        baseline_cost_usd: Any = _UNSET,
    ) -> bool:
        """Apply a caller-supplied outcome and optional measured costs later."""
        if not isinstance(success, bool):
            raise ValueError("success must be a bool")
        assignments = ["success=?", "outcome_updated_at=?"]
        values: list[Any] = [int(success), datetime.now(timezone.utc).isoformat()]
        for name, value in (
            ("cost_usd", cost_usd),
            ("baseline_cost_usd", baseline_cost_usd),
        ):
            if value is not _UNSET:
                _nonnegative(value, name)
                assignments.append(f"{name}=?")
                values.append(value)
        values.extend([tenant, request_id])
        with self._lock, self._db:
            cursor = self._db.execute(
                f"UPDATE requests SET {', '.join(assignments)} "
                "WHERE tenant=? AND request_id=?",
                values,
            )
        return cursor.rowcount == 1

    def cached(self, tenant: str, provider: str, body: dict) -> dict | None:
        """Use the latest resolved outcome; later failures invalidate old success.

        An unresolved newer request is not evidence that a previously verified
        answer failed. Explicit late outcome reversals are ordered by update time.
        """
        with self._lock:
            row = self._db.execute(
                """SELECT * FROM requests
                   WHERE tenant=? AND fingerprint=? AND success IS NOT NULL
                   ORDER BY outcome_updated_at DESC, created_at DESC LIMIT 1""",
                (tenant, request_fingerprint(provider, body)),
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
        """Report known ledger amounts, never extrapolate missing costs."""
        with self._lock:
            rows = self._db.execute(
                "SELECT success,cost_usd,baseline_cost_usd,cached_from "
                "FROM requests WHERE tenant=?",
                (tenant,),
            ).fetchall()
        comparable = [
            row
            for row in rows
            if row["success"] == 1
            and row["cost_usd"] is not None
            and row["baseline_cost_usd"] is not None
        ]
        verified_cost = sum(row["cost_usd"] for row in comparable)
        verified_baseline = sum(row["baseline_cost_usd"] for row in comparable)
        known_cost = sum(
            row["cost_usd"] for row in rows if row["cost_usd"] is not None
        )
        successes = sum(row["success"] == 1 for row in rows)
        return {
            "tenant": tenant,
            "requests": len(rows),
            "verified_successes": successes,
            "failures": sum(row["success"] == 0 for row in rows),
            "pending_outcomes": sum(row["success"] is None for row in rows),
            "cache_hits": sum(row["cached_from"] is not None for row in rows),
            "known_cost_total_usd": known_cost,
            "known_cost_per_verified_success_usd": (
                known_cost / successes if successes else None
            ),
            "unknown_cost_requests": sum(row["cost_usd"] is None for row in rows),
            "comparable_verified_requests": len(comparable),
            "verified_cost_usd": verified_cost,
            "verified_baseline_cost_usd": verified_baseline,
            "verified_savings_usd": verified_baseline - verified_cost,
            "savings_fraction": (
                1 - verified_cost / verified_baseline if verified_baseline > 0 else None
            ),
            "scope": "development ledger; outcomes and baseline costs are caller-attested",
            "savings_scope": (
                "comparable successful requests only; failed and pending requests "
                "are excluded from this subtotal, so it is not a billing amount"
            ),
        }
