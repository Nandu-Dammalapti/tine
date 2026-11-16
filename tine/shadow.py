"""Shadow mode: run a frozen policy beside the customer's own model and report.

In shadow mode nothing the customer sees changes. Their request is answered by the
frontier model as before. For a sample of those requests the proxy then runs the
cheaper stages of a frozen policy (``tine.policy``) in the background, on the same
provider account, and records what the policy would have done: whether it would have
returned a cheaper answer, whether that answer equals the one the frontier gave, and
what each path cost. The report built from those records is the free savings report
that precedes paid routing.

What the report can and cannot say:

- It compares the policy's accepted answer with the frontier's answer for the same
  request. Equal answers are evidence for exact-answer work (a label, a number, a
  choice); they are not a test of correctness, and for free-form text two right
  answers rarely match. The quality guarantee comes from the policy's own
  certification, not from this comparison.
- Shadow calls cost money on the customer's account. That spend is reported and can
  be capped. Projected savings are computed only from requests whose costs are all
  known.
- Only stages with built-in checks (agreement, none) can be shadowed here. A check
  that needs the customer's repository or a judge call is not run by the proxy.

No prompt or response text is stored by this module; the ledger's request record
already holds the customer's request and the frontier's response.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any

from tine.policy import BUILT_IN_CHECKS, same_answer, validate_policy

SHADOW_SCOPE = (
    "sampled requests answered by the frontier model; the policy's cheaper stages were run in the "
    "background and never returned to the caller. Equal answers are not proof of correctness. "
    "Projected figures cover only requests whose costs are all known and are not a billing amount."
)


def sampled(request_id: str, rate: float) -> bool:
    """A fixed share of requests, decided by the request id alone."""
    if rate <= 0:
        return False
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def response_text(provider: str, payload: Any) -> str | None:
    """The text a chat response returned, or None if there is none."""
    try:
        if provider == "openai":
            text = payload["choices"][0]["message"]["content"]
        else:
            text = "".join(block["text"] for block in payload["content"] if block.get("type") == "text")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return text if isinstance(text, str) and text.strip() else None


def extract_answer(provider: str, payload: Any, pattern: str | None = None) -> str | None:
    """The comparable final answer of a response. With ``pattern`` it is the last
    match's first group (or the whole match); without, the whole text."""
    text = response_text(provider, payload)
    if text is None or not pattern:
        return text
    found = None
    for found in re.finditer(pattern, text):
        pass
    if found is None:
        return None
    answer = found.group(1) if found.groups() else found.group(0)
    return answer if answer and answer.strip() else None


def validate_shadow_policy(policy: dict, frontier: str) -> dict:
    """A policy the proxy can shadow: valid, for this frontier, built-in checks only."""
    validate_policy(policy)
    if policy["frontier"] != frontier:
        raise ValueError("A shadow policy's frontier must be the provider's configured frontier model")
    unsupported = sorted({stage["check"] for stage in policy["stages"]} - set(BUILT_IN_CHECKS))
    if unsupported:
        raise ValueError(f"Shadow mode runs agreement stages only; this policy needs {unsupported}")
    return policy


async def shadow_stages(policy: dict, call) -> dict:
    """Run the cheaper stages of ``policy`` and say what it would have done.

    ``call(model)`` is awaited and returns a mapping with ``answer`` and ``cost_usd``.
    The frontier is not called: in shadow mode its answer already exists. A model that
    fails has no answer, so its stage does not pass. A model that appears in two
    stages is called once.
    """
    seen: dict[str, dict | None] = {}
    calls = []

    async def attempt(model: str):
        if model in seen:
            return seen[model]
        try:
            result = await call(model)
        except Exception as error:  # a shadow failure is a data point, never an outage
            calls.append({"model": model, "cost_usd": None, "error": type(error).__name__})
            seen[model] = None
            return None
        calls.append({"model": model, "cost_usd": result.get("cost_usd"), "error": None})
        seen[model] = result
        return result

    for index, stage in enumerate(policy["stages"]):
        results = await asyncio.gather(*(attempt(model) for model in stage["models"]))
        if any(result is None for result in results):
            continue
        if stage["check"] == "none" or same_answer([result.get("answer") for result in results]):
            return {"accepted": True, "stage": index, "served_by": stage["models"][0],
                    "answer": results[0].get("answer"), "calls": calls}
    return {"accepted": False, "stage": None, "served_by": None, "answer": None, "calls": calls}


def _known_sum(values) -> float | None:
    values = list(values)
    if any(value is None for value in values):
        return None
    total = math.fsum(values)
    return total if math.isfinite(total) else None


class ShadowStore:
    """Shadow results beside the ledger's requests, in the same database."""

    def __init__(self, ledger) -> None:
        self._ledger = ledger
        with ledger._lock, ledger._db:
            ledger._db.execute(
                """CREATE TABLE IF NOT EXISTS shadow_runs (
                    request_id TEXT PRIMARY KEY,
                    tenant TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    policy_sha256 TEXT NOT NULL,
                    accepted INTEGER NOT NULL CHECK(accepted IN (0, 1)),
                    stage INTEGER,
                    served_by TEXT,
                    matches_frontier INTEGER CHECK(matches_frontier IN (0, 1) OR matches_frontier IS NULL),
                    calls_json TEXT NOT NULL,
                    shadow_cost_usd REAL,
                    frontier_cost_usd REAL
                )"""
            )
            ledger._db.execute("CREATE INDEX IF NOT EXISTS shadow_tenant ON shadow_runs(tenant)")

    def record(self, *, request_id: str, tenant: str, provider: str, policy: dict, outcome: dict,
               frontier_answer: str | None, frontier_cost_usd: float | None) -> None:
        matches = None
        if outcome["accepted"]:
            matches = int(frontier_answer is not None and same_answer([outcome["answer"], frontier_answer]))
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        with self._ledger._lock, self._ledger._db:
            self._ledger._db.execute(
                "INSERT INTO shadow_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id, tenant, provider, datetime.now(timezone.utc).isoformat(), digest,
                 int(outcome["accepted"]), outcome["stage"], outcome["served_by"], matches,
                 json.dumps(outcome["calls"], allow_nan=False),
                 _known_sum(entry["cost_usd"] for entry in outcome["calls"]), frontier_cost_usd))

    def spend(self, tenant: str) -> float:
        """Known shadow spend so far. Calls of unknown cost are not counted, so a cap
        on this figure needs prices configured for every stage model."""
        with self._ledger._lock:
            rows = self._ledger._db.execute(
                "SELECT calls_json FROM shadow_runs WHERE tenant=?", (tenant,)).fetchall()
        return math.fsum(entry["cost_usd"] for row in rows for entry in json.loads(row["calls_json"])
                         if entry["cost_usd"] is not None)

    def report(self, tenant: str) -> dict[str, Any]:
        with self._ledger._lock:
            rows = self._ledger._db.execute(
                "SELECT accepted,matches_frontier,shadow_cost_usd,frontier_cost_usd,calls_json "
                "FROM shadow_runs WHERE tenant=?", (tenant,)).fetchall()
        accepted = [row for row in rows if row["accepted"]]
        matching = sum(row["matches_frontier"] == 1 for row in accepted)
        known = [row for row in rows if row["shadow_cost_usd"] is not None and row["frontier_cost_usd"] is not None]
        frontier_cost = math.fsum(row["frontier_cost_usd"] for row in known)
        # What the policy would have cost: every cheaper call it made, plus the
        # frontier where no cheaper answer was accepted.
        policy_cost = math.fsum(row["shadow_cost_usd"] + (0.0 if row["accepted"] else row["frontier_cost_usd"])
                                for row in known)
        calls = [entry for row in rows for entry in json.loads(row["calls_json"])]
        return {
            "tenant": tenant, "shadowed_requests": len(rows),
            "policy_would_accept": len(accepted),
            "policy_would_accept_share": len(accepted) / len(rows) if rows else None,
            "accepted_matching_frontier": matching,
            "accepted_matching_frontier_share": matching / len(accepted) if accepted else None,
            "cost_known_requests": len(known),
            "unknown_cost_requests": len(rows) - len(known),
            "frontier_cost_usd": frontier_cost if known else None,
            "projected_policy_cost_usd": policy_cost if known else None,
            "projected_savings_usd": frontier_cost - policy_cost if known else None,
            "projected_savings_fraction": 1 - policy_cost / frontier_cost if known and frontier_cost > 0 else None,
            "shadow_spend_usd": math.fsum(entry["cost_usd"] for entry in calls if entry["cost_usd"] is not None),
            "shadow_calls": len(calls),
            "failed_shadow_calls": sum(entry["error"] is not None for entry in calls),
            "scope": SHADOW_SCOPE,
        }
