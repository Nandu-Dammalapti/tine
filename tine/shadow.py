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
- Shadow calls cost money on the customer's account. That spend is reported and
  capped. The cap counts known costs: it is held in memory per tenant, a run in
  flight reserves room under it, and calls whose cost is not known are counted
  and stop shadowing when there are too many. Projected savings are computed only
  from requests whose costs are all known.
- A stage with a named check (tests, a judge) is shadowed only when the proxy has
  been given a way to run that check (``tine.runtime_checks``). For such stages the
  answer is not compared with the frontier's: two right patches rarely match.

No prompt or response text is stored by this module; the ledger's request record
already holds the customer's request and the frontier's response.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any

from tine.policy import BUILT_IN_CHECKS, known_cost, run_stages_async, same_answer, validate_policy

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


class NoAnswer(Exception):
    """A provider reply that was paid for and holds nothing to serve or compare."""

    def __init__(self, cost_usd: float | None) -> None:
        super().__init__("the reply holds no answer")
        self.cost_usd = cost_usd


def _blocks(provider: str, payload: Any) -> tuple[list, list, Any]:
    """The text parts, tool calls and stop reason of a chat reply, however odd its shape."""
    try:
        if provider == "openai":
            choice = payload["choices"][0]
            message = choice["message"]
            text = message.get("content")
            tools = [*(message.get("tool_calls") or []), *([message["function_call"]] if message.get("function_call") else [])]
            return ([text] if isinstance(text, str) and text.strip() else []), tools, choice.get("finish_reason")
        content = [block for block in payload["content"] if isinstance(block, dict)]
        texts = [block["text"] for block in content
                 if block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"].strip()]
        tools = [block for block in content if str(block.get("type", "")).endswith("tool_use")]
        return texts, tools, payload.get("stop_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return [], [], None


def has_answer(provider: str, payload: Any) -> bool:
    """Whether a 200 reply is an answer at all: text or a tool call, and not withheld.

    An empty reply, an error body sent with status 200, a content-filtered reply and
    a refusal are provider failures. They are not answers a policy was certified on,
    so no stage may serve them, whatever its check.
    """
    texts, tools, stop = _blocks(provider, payload)
    return bool(texts or tools) and stop not in ("content_filter", "refusal")


def plain_finish(provider: str, payload: Any) -> bool:
    """Whether a reply is a finished text answer: no tool call, and not cut off.

    Agreement was certified on final answers. Two replies with the same words and
    different tool calls did not agree on what to do, and a reply cut off by the
    token limit is not a final answer, so neither has an answer to compare.
    """
    _, tools, stop = _blocks(provider, payload)
    return not tools and stop in (None, "stop", "end_turn", "stop_sequence")


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


def validate_shadow_policy(policy: dict, frontier: str, available_checks=()) -> dict:
    """A policy the proxy can run: valid, for this frontier, and every named check
    it needs is one the proxy has been configured to run."""
    validate_policy(policy)
    if policy["frontier"] != frontier:
        raise ValueError("A policy's frontier must be the provider's configured frontier model")
    if policy.get("answer_pattern") is not None:
        try:
            re.compile(policy["answer_pattern"])
        except (re.error, TypeError):
            raise ValueError("A policy's answer_pattern must be a valid regular expression") from None
    unsupported = sorted({stage["check"] for stage in policy["stages"]} - set(BUILT_IN_CHECKS)
                         - set(available_checks))
    if unsupported:
        raise ValueError(f"No check is configured for {unsupported}; the proxy runs agreement stages "
                         "and the named checks it has been given")
    return policy


async def shadow_stages(policy: dict, call, checks: dict | None = None) -> dict:
    """Run the cheaper stages of ``policy`` and say what it would have done.

    ``call(model)`` is awaited and returns a mapping with ``answer`` and ``cost_usd``.
    The frontier is not called: in shadow mode its answer already exists. A model that
    fails has no answer, so its stage does not pass. A model that appears in two
    stages is called once.
    """
    outcome = await run_stages_async(policy, call, checks=checks)
    result = outcome["result"] or {}
    return {"accepted": outcome["accepted"], "stage": outcome["stage"], "served_by": outcome["served_by"],
            "answer": result.get("answer"), "calls": outcome["calls"], "checks": outcome["checks"],
            "comparable": outcome["check"] in BUILT_IN_CHECKS}


def _known_sum(values) -> float | None:
    values = list(values)
    if any(value is None for value in values):
        return None
    try:
        total = math.fsum(values)
    except OverflowError:
        return None
    return total if math.isfinite(total) else None


def shadow_entries(outcome: dict) -> list[dict]:
    """Every call and check of one shadow run as costed entries, in the form they are
    stored. A cost that is not a usable amount is stored as unknown."""
    entries = [*outcome["calls"], *({**run, "model": f"check:{run['check']}"} for run in outcome.get("checks", []))]
    return [{**entry, "cost_usd": known_cost(entry.get("cost_usd"))} for entry in entries]


def _known_part(entry: dict) -> float:
    """What is known of an entry's cost: all of it, the known steps of a check whose
    total is unknown, or nothing."""
    cost = entry.get("cost_usd")
    return cost if cost is not None else (known_cost(entry.get("known_cost_usd")) or 0.0)


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
        # Per tenant: what the spending cap has counted so far (see ``admit``).
        self._budgets: dict[str, dict] = {}

    def record(self, *, request_id: str, tenant: str, provider: str, policy: dict, outcome: dict,
               frontier_answer: str | None, frontier_cost_usd: float | None) -> None:
        matches = None
        if outcome["accepted"] and outcome.get("comparable", True):
            matches = int(frontier_answer is not None and same_answer([outcome["answer"], frontier_answer]))
        spent = shadow_entries(outcome)
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        with self._ledger._lock, self._ledger._db:
            self._ledger._db.execute(
                "INSERT INTO shadow_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id, tenant, provider, datetime.now(timezone.utc).isoformat(), digest,
                 int(outcome["accepted"]), outcome["stage"], outcome["served_by"], matches,
                 json.dumps(spent, allow_nan=False),
                 _known_sum(entry["cost_usd"] for entry in spent), frontier_cost_usd))

    def _runs(self, tenant: str) -> list[list[dict]]:
        """The stored entries of every shadow run of a tenant. Reads the whole table."""
        with self._ledger._lock:
            rows = self._ledger._db.execute(
                "SELECT calls_json FROM shadow_runs WHERE tenant=?", (tenant,)).fetchall()
        return [json.loads(row["calls_json"]) for row in rows]

    def spend(self, tenant: str) -> float:
        """Known shadow spend recorded so far. Calls of unknown cost are not in it."""
        return math.fsum(_known_part(entry) for run in self._runs(tenant) for entry in run)

    def _budget(self, tenant: str) -> dict:
        """The cap's running figures for a tenant: read from the table once, then kept
        in memory, so a request does not pay for a scan that grows with the table."""
        if tenant not in self._budgets:
            runs = self._runs(tenant)
            self._budgets[tenant] = {
                "spent": math.fsum(_known_part(entry) for run in runs for entry in run),
                "unknown": sum(entry["cost_usd"] is None for run in runs for entry in run),
                "largest": max((math.fsum(_known_part(entry) for entry in run) for run in runs), default=None),
                "running": 0}
        return self._budgets[tenant]

    def state(self, tenant: str, cap: float, max_unknown_cost_calls: int) -> str:
        """Whether a new shadow run may start for a tenant, in words for the report."""
        budget = self._budget(tenant)
        if budget["unknown"] > max_unknown_cost_calls:
            return "stopped: too many shadow calls of unknown cost"
        if budget["spent"] >= cap:
            return "stopped: the spending cap is reached"
        return "on"

    def admit(self, tenant: str, cap: float, max_unknown_cost_calls: int) -> bool:
        """Reserve room under the cap for one more shadow run, or refuse it.

        A run's cost is known only when it ends, so a run in flight is held to cost
        what the tenant's most expensive run so far did. Until one run has finished
        there is nothing to go by and one run at a time is let through. The run that
        crosses the cap still completes, so spend can pass the cap by about one run,
        and by more only if runs in flight cost more than any run before them.
        Every admitted run must be followed by ``settle``.
        """
        budget = self._budget(tenant)
        if self.state(tenant, cap, max_unknown_cost_calls) != "on":
            return False
        if budget["running"] and budget["largest"] is None:
            return False
        if budget["spent"] + budget["running"] * (budget["largest"] or 0.0) >= cap:
            return False
        budget["running"] += 1
        return True

    def settle(self, tenant: str, entries: list[dict]) -> None:
        """End an admitted run and count what it spent, whether or not its row could be
        written: the money was spent either way."""
        budget = self._budget(tenant)
        cost = math.fsum(_known_part(entry) for entry in entries)
        budget["running"] -= 1
        budget["spent"] += cost
        budget["unknown"] += sum(entry["cost_usd"] is None for entry in entries)
        if entries:
            budget["largest"] = max(budget["largest"] or 0.0, cost)

    def report(self, tenant: str) -> dict[str, Any]:
        with self._ledger._lock:
            rows = self._ledger._db.execute(
                "SELECT accepted,matches_frontier,shadow_cost_usd,frontier_cost_usd,calls_json "
                "FROM shadow_runs WHERE tenant=?", (tenant,)).fetchall()
        accepted = [row for row in rows if row["accepted"]]
        compared = [row for row in accepted if row["matches_frontier"] is not None]
        matching = sum(row["matches_frontier"] == 1 for row in compared)
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
            "accepted_compared_with_frontier": len(compared),
            "accepted_matching_frontier": matching,
            "accepted_matching_frontier_share": matching / len(compared) if compared else None,
            "cost_known_requests": len(known),
            "unknown_cost_requests": len(rows) - len(known),
            "frontier_cost_usd": frontier_cost if known else None,
            "projected_policy_cost_usd": policy_cost if known else None,
            "projected_savings_usd": frontier_cost - policy_cost if known else None,
            "projected_savings_fraction": 1 - policy_cost / frontier_cost if known and frontier_cost > 0 else None,
            # A subtotal of known costs. It is zero, not unknown, when nothing is known:
            # read it with the count of calls and checks whose cost is not known.
            "shadow_spend_usd": math.fsum(_known_part(entry) for entry in calls),
            "unknown_cost_shadow_calls": sum(entry["cost_usd"] is None for entry in calls),
            "shadow_calls": len(calls),
            "failed_shadow_calls": sum(entry["error"] is not None for entry in calls),
            "scope": SHADOW_SCOPE,
        }
