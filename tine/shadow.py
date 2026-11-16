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
  and stop shadowing when there are both more of them than a set number and more
  than a set share of all shadow calls and checks. Projected savings are computed
  only from requests whose costs are all known.
- Not every sampled request is shadowed. A sampled request is skipped when too many
  runs are in flight or the spending cap has no room for it yet, or when shadowing
  has stopped; a streamed request is never shadowed. Each of these is counted per
  tenant, and the report gives the share of sampled requests that were shadowed, so
  the projection says what it left out.
- When too many accepted cheaper answers differ from the frontier's answer (as a
  share of the shadowed requests, more than the policy's quality limit, on at least
  ``DRIFT_MIN_COMPARED`` compared answers), the report carries a warning: such
  traffic cannot confirm the certification. A different answer is not always a wrong one, so this is a reason
  to grade a sample, not a measured loss.
- A stage with a named check (tests, a judge) is shadowed only when the proxy has
  been given a way to run that check (``tine.runtime_checks``). For such stages the
  answer is not compared with the frontier's: two right patches rarely match.
- A sampled request that carries tool definitions is not run through a policy with
  agreement stages: cheaper models cannot agree on tool calls. It is recorded as
  sent to the frontier, with no shadow calls and the reason, as routing would do.

No prompt or response text is stored by this module; the ledger's request record
already holds the customer's request and the frontier's response.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tine.policy import BUILT_IN_CHECKS, known_cost, run_stages_async, same_answer, validate_policy

SHADOW_SCOPE = (
    "sampled requests answered by the frontier model; the policy's cheaper stages were run in the "
    "background and never returned to the caller. Equal answers are not proof of correctness. "
    "Projected figures cover only requests whose costs are all known and are not a billing amount."
)
# Below this many accepted answers compared with the frontier's, the share that differ
# says too little to warn on.
DRIFT_MIN_COMPARED = 30
# What became of a sampled request that has no shadow row, counted per tenant
# (``ShadowStore.count``). A shadowed request is counted by its row.
COUNTED = ("skipped_busy", "skipped_stopped", "failed", "streamed")


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


# An answer pattern that reads a whole reply as its answer, for replies that are the
# answer and nothing else (a recorded answer, a constrained label). It reads exactly
# what no pattern reads, but is a reader chosen on purpose.
WHOLE_REPLY_PATTERN = r"(?s).+"
TOOLS_SKIP_REASON ="tools: the request carries tool definitions, and agreement cannot compare tool calls"


def carries_tools(body: Any) -> bool:
    """Whether a request defines tools or functions the model may call."""
    return isinstance(body, dict) and any(body.get(key) for key in ("tools", "functions"))


def skips_for_tools(policy: dict, body: Any) -> bool:
    """Whether a request goes straight to the frontier instead of through ``policy``:
    it carries tool definitions and the policy has an agreement stage. Agreement was
    certified on final answers, and replies that call tools have none to compare. The
    rest of the chain is not run either: without a stage it is not the chain that was
    certified."""
    return carries_tools(body) and any(stage["check"] == "agreement" for stage in policy["stages"])


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


def _columns(ledger) -> set[str]:
    return {row["name"] for row in ledger._db.execute("PRAGMA table_info(shadow_runs)")}


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
                    frontier_cost_usd REAL,
                    skip_reason TEXT
                )"""
            )
            ledger._db.execute("CREATE INDEX IF NOT EXISTS shadow_tenant ON shadow_runs(tenant)")
            if "skip_reason" not in _columns(ledger):
                ledger._db.execute("ALTER TABLE shadow_runs ADD COLUMN skip_reason TEXT")
            ledger._db.execute(
                """CREATE TABLE IF NOT EXISTS shadow_counts (
                    tenant TEXT NOT NULL,
                    name TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (tenant, name)
                )""")
        # Per tenant: what the spending cap has counted so far (see ``admit``).
        self._budgets: dict[str, dict] = {}

    def record(self, *, request_id: str, tenant: str, provider: str, policy: dict, outcome: dict,
               frontier_answer: str | None, frontier_cost_usd: float | None, skip_reason: str | None = None) -> None:
        """Store one shadow run. ``skip_reason`` marks a request the policy would have
        sent to the frontier without trying its stages; ``outcome`` then has no calls."""
        matches = None
        if outcome["accepted"] and outcome.get("comparable", True):
            matches = int(frontier_answer is not None and same_answer([outcome["answer"], frontier_answer]))
        spent = shadow_entries(outcome)
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        with self._ledger._lock, self._ledger._db:
            self._ledger._db.execute(
                "INSERT INTO shadow_runs (request_id, tenant, provider, created_at, policy_sha256, accepted, stage, "
                "served_by, matches_frontier, calls_json, shadow_cost_usd, frontier_cost_usd, skip_reason) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id, tenant, provider, datetime.now(timezone.utc).isoformat(), digest,
                 int(outcome["accepted"]), outcome["stage"], outcome["served_by"], matches,
                 json.dumps(spent, allow_nan=False),
                 _known_sum(entry["cost_usd"] for entry in spent), frontier_cost_usd, skip_reason))

    def count(self, tenant: str, name: str) -> None:
        """Count one sampled request that has no row of its own: it was skipped
        (``skipped_busy``, ``skipped_stopped``), its run failed before its row was
        written (``failed``), or it was streamed (``streamed``)."""
        if name not in COUNTED:
            raise ValueError(f"Not a shadow count: {name!r}")
        with self._ledger._lock, self._ledger._db:
            self._ledger._db.execute(
                "INSERT INTO shadow_counts VALUES (?,?,1) ON CONFLICT(tenant, name) DO UPDATE SET count=count+1",
                (tenant, name))

    def counts(self, tenant: str) -> dict[str, int] | None:
        """The tenant's counts, or None for a ledger written before they were kept."""
        try:
            with self._ledger._lock:
                rows = self._ledger._db.execute(
                    "SELECT name, count FROM shadow_counts WHERE tenant=?", (tenant,)).fetchall()
        except sqlite3.OperationalError:  # a read-only copy of an older ledger has no such table
            return None
        found = {row["name"]: row["count"] for row in rows}
        return {name: found.get(name, 0) for name in COUNTED}

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
                "calls": sum(len(run) for run in runs),
                "why": Counter(_why_unknown(entry) for run in runs for entry in run if entry["cost_usd"] is None),
                "largest": _largest(math.fsum(_known_part(entry) for entry in run) for run in runs),
                "running": 0}
        return self._budgets[tenant]

    def state(self, tenant: str, cap: float, max_unknown_cost_calls: int,
              max_unknown_cost_share: float = 0.0) -> str:
        """Whether a new shadow run may start for a tenant, in words for the report.

        Calls and checks of unknown cost stop shadowing only when there are more than
        ``max_unknown_cost_calls`` of them and they are also more than
        ``max_unknown_cost_share`` of all the tenant's shadow calls and checks, so
        that occasional failures (a flaky CI run, a provider's bad minute) do not end
        shadow mode. The figures are read from the shadow rows, so a stop holds
        across restarts; the stop reason names the most common failures.
        """
        budget = self._budget(tenant)
        unknown, calls = budget["unknown"], budget["calls"]
        if unknown > max_unknown_cost_calls and unknown > max_unknown_cost_share * calls:
            causes = "; ".join(f"{why} ({times:,})" for why, times in budget["why"].most_common(3))
            return (f"stopped: too many shadow calls of unknown cost: {unknown:,} of {calls:,} calls and checks "
                    f"({unknown / calls:.1%}), more than {max_unknown_cost_calls:,} and more than "
                    f"{max_unknown_cost_share:.1%} of them. Most common: {causes}")
        if budget["spent"] >= cap:
            return "stopped: the spending cap is reached"
        return "on"

    def admit(self, tenant: str, cap: float, max_unknown_cost_calls: int,
              max_unknown_cost_share: float = 0.0) -> bool:
        """Reserve room under the cap for one more shadow run, or refuse it.

        A run's cost is known only when it ends, so a run in flight is held to cost
        what the tenant's most expensive run so far did. Until one run has finished
        there is nothing to go by and one run at a time is let through. A run with no
        known cost (no model call, or none whose cost is known) is nothing to go by
        either: a zero would let every run in flight through. The run that
        crosses the cap still completes, so spend can pass the cap by about one run,
        and by more only if runs in flight cost more than any run before them.
        Every admitted run must be followed by ``settle``.
        """
        budget = self._budget(tenant)
        if self.state(tenant, cap, max_unknown_cost_calls, max_unknown_cost_share) != "on":
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
        budget["calls"] += len(entries)
        budget["why"].update(_why_unknown(entry) for entry in entries if entry["cost_usd"] is None)
        budget["largest"] = _largest([cost] if budget["largest"] is None else [budget["largest"], cost])

    def report(self, tenant: str, epsilon: float | None = None,
               min_compared: int = DRIFT_MIN_COMPARED) -> dict[str, Any]:
        """The tenant's shadow figures. ``epsilon`` is the quality limit of the policy
        being shadowed; with it, the share of accepted answers that differ from the
        frontier's is checked against that limit (``drift``)."""
        with self._ledger._lock:
            # A ledger opened read-only may predate the column: nothing in it was skipped.
            skipped = "skip_reason" if "skip_reason" in _columns(self._ledger) else "NULL AS skip_reason"
            rows = self._ledger._db.execute(
                f"SELECT accepted,matches_frontier,shadow_cost_usd,frontier_cost_usd,calls_json,{skipped} "
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
            # Included in shadowed_requests, as requests the policy sends to the frontier.
            "sent_to_frontier_for_tools": sum(row["skip_reason"] == TOOLS_SKIP_REASON for row in rows),
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
            # Checks that rejected for a stated reason rather than on the answer, such as
            # a request too long for the judge to read (its stage then calls no model).
            "check_rejection_reasons": dict(sorted(Counter(
                entry["reason"] for entry in calls if isinstance(entry.get("reason"), str)).items())),
            **coverage(len(rows), self.counts(tenant)),
            # Requests the policy sends to the frontier lose nothing against it; accepted
            # answers that were not compared cannot be told either way and are left out.
            **drift(len(compared), len(compared) - matching, epsilon, min_compared,
                    requests=len(rows) - (len(accepted) - len(compared))),
            "scope": SHADOW_SCOPE,
        }


def _largest(costs) -> float | None:
    """The largest known cost of a shadow run, from runs that cost something, or None."""
    return max((cost for cost in costs if cost > 0), default=None)


def _why_unknown(entry: dict) -> str:
    """Why one call or check has no known cost, in a few words for a stop reason."""
    if entry.get("status") is not None:
        why = f"HTTP {entry['status']}"
    elif entry.get("error"):
        why = str(entry["error"])
    else:
        why = "no usage or no price in the reply"
    return f"{entry.get('model')}: {why}"


def coverage(shadowed: int, counts: dict[str, int] | None) -> dict[str, Any]:
    """What became of the sampled requests: how many were shadowed, and why the rest
    were not.

    ``sampled_requests`` counts the sampled requests a shadow run could have taken
    (answered by the frontier without streaming) once their run ended or was skipped:
    the ``shadowed`` ones, which have a row, and the skipped and failed ones, which
    are counted. ``shadow_coverage`` is the share of them that were shadowed. Busy
    periods lose more of their sample, so a coverage below one means the projection
    leans to quieter ones. Streamed requests are counted apart: they were sampled,
    but shadow mode cannot run them. The counts start when a ledger is first opened
    by a proxy that keeps them; all are None for a ledger that has none to read.
    """
    if counts is None:
        return dict.fromkeys(("sampled_requests", "skipped_busy", "skipped_stopped", "failed_shadow_runs",
                              "streamed_not_shadowed", "shadow_coverage"))
    sampled = shadowed + counts["skipped_busy"] + counts["skipped_stopped"] + counts["failed"]
    return {"sampled_requests": sampled, "skipped_busy": counts["skipped_busy"],
            "skipped_stopped": counts["skipped_stopped"], "failed_shadow_runs": counts["failed"],
            "streamed_not_shadowed": counts["streamed"],
            "shadow_coverage": shadowed / sampled if sampled else None}


def drift(compared: int, differing: int, epsilon: float | None, min_compared: int = DRIFT_MIN_COMPARED,
          *, requests: int | None = None) -> dict:
    """Whether so many accepted answers differ from the frontier's that this traffic
    cannot confirm the policy's certification.

    The policy is certified to lose at most ``epsilon`` of all tasks against the
    frontier. An accepted answer equal to the frontier's loses nothing against it,
    and neither does a request the policy sends to the frontier. So the differing
    answers are counted as a share of ``requests``: the shadowed requests whose
    answer was compared with the frontier's or sent to the frontier (``compared``
    when not given). A share at or below ``epsilon`` is consistent with the
    certification, and one above it cannot confirm it. The share among accepted
    answers alone is larger by about one over the acceptance rate, and is reported
    beside it, not checked. Cheaper models that agree on wrong answers push the share
    up while they push the projected saving up too. A different answer is not always
    a wrong one (the frontier can be the one that is wrong), so the warning asks for
    grading; it is not a measured loss.
    """
    requests = compared if requests is None else requests
    share = differing / requests if requests else None
    result = {"accepted_differing_from_frontier": differing,
              "accepted_differing_from_frontier_share": differing / compared if compared else None,
              "drift_requests": requests, "drift_share": share,
              "drift_epsilon": epsilon, "drift_min_compared": min_compared, "drift_warning": None}
    counted = (f"{share:.2%} of the {requests:,} shadowed requests whose answer was compared with the frontier's "
               "or sent to the frontier") if share is not None else None
    if epsilon is None:
        result["drift_check"] = "not checked: the policy's quality limit was not given"
    elif compared < min_compared:
        result["drift_check"] = (f"not checked yet: {compared:,} accepted answers compared with the frontier's, "
                                 f"{min_compared:,} needed")
    elif share > epsilon:
        result["drift_check"] = "warning"
        result["drift_warning"] = (
            f"Cannot confirm the certification on this traffic: {differing:,} of {compared:,} accepted cheaper "
            f"answers differ from the frontier model's answer, {counted}, more than the policy's quality limit "
            f"of {epsilon:.2%}. A different answer is not always a wrong one; grade a sample of these requests "
            "before relying on the policy or its projected saving.")
    else:
        result["drift_check"] = (f"within the limit: {differing:,} of {compared:,} accepted answers differ from the "
                                 f"frontier's, {counted}, at most the policy's quality limit of {epsilon:.2%}")
    return result
