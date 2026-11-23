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
  certification, not from this comparison. Answers are compared the way the policy's
  agreement compares them: exactly, or by value with ``answer_compare: "math"``
  (``tine.policy.canonical_answer``).
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
- Shadow calls share the provider account's rate limit with the customer's real
  traffic. When a provider refuses a call for its rate limit or load (429, 529), on a
  shadow run or on real traffic, shadow mode pauses on that provider (``ShadowLimits``)
  and the sampled requests that arrive meanwhile are skipped and counted. A run that
  was refused part way is stored apart and left out of the projection: it does not show
  what the policy would do.
- The proxy cannot see right or wrong, only whether two answers are the same. The
  drift check (``drift``) counts the accepted cheaper answers that differ from the
  frontier's, as a share of the shadowed requests the policy was run on and as a
  share of the accepted answers, and compares each share with what the policy's
  certification allows for: its quality limit plus the share of answers that
  differed without a loss when it was frozen (the policy's ``drift_baseline``), or
  plus ``DRIFT_DEFAULT_ALLOWANCE`` for a policy file that records none. When a share
  is clearly above that, on at least ``DRIFT_MIN_COMPARED`` compared answers, the
  report carries a warning: such traffic cannot confirm the certification. When it
  is clearly above the share that differed on the graded tasks, but not above what
  the certification allows for, the report carries a notice. A different answer is
  not always a wrong one, so both are a reason to grade a sample, not a measured
  loss, and a cheaper answer equal to a wrong frontier answer is never seen.
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
import time
from collections import Counter, deque
from datetime import datetime, timezone
from itertools import accumulate
from typing import Any

from tine.policy import BUILT_IN_CHECKS, answer_mode, known_cost, run_stages_async, same_answer, validate_policy

SHADOW_SCOPE = (
    "sampled requests answered by the frontier model; the policy's cheaper stages were run in the "
    "background and never returned to the caller. Equal answers are not proof of correctness. "
    "Projected figures cover only requests whose costs are all known and are not a billing amount."
)
# Below this many accepted answers compared with the frontier's, the share that differ
# says too little to warn on.
DRIFT_MIN_COMPARED = 30
# The drift check (``drift``). The share of differing answers it allows for is the
# policy's quality limit plus an upper bound, at this confidence, on the share that
# differed without a loss when the policy was frozen. A policy file with no such
# baseline gets a fixed number of points of its requests instead, and a fixed share of
# its accepted answers. The check warns when the evidence against "at most that share
# differ" reaches the level needed; on traffic within the allowance a warning then
# shows with a chance of at most one over that level (``drift_evidence``). The evidence
# is also weighed afresh from each multiple of the block, so that a change late in a
# history shows, also in a run of a thousand requests.
DRIFT_BASELINE_CONFIDENCE = 0.95
DRIFT_DEFAULT_ALLOWANCE = 0.10
DRIFT_DEFAULT_ALLOWANCE_ACCEPTED = 1 / 3
DRIFT_EVIDENCE_NEEDED = 20.0
DRIFT_BLOCK = 250
# Evidence is reported up to this; beyond it the figure says nothing more.
DRIFT_EVIDENCE_CAP = 1e12
# What became of a sampled request that has no shadow row, counted per tenant
# (``ShadowStore.count``). A shadowed request is counted by its row.
COUNTED = ("skipped_busy", "skipped_stopped", "failed", "streamed", "skipped_rate_limited", "skipped_rate_cap")
# A provider reply that says the account is at its rate limit (429) or the provider is
# overloaded (529). The provider refused the request before doing any work and does
# not bill it.
LIMIT_STATUSES = (429, 529)
# How long shadow mode pauses on a provider that refused a call without saying how long
# to wait: the first pause, doubled for each pause in a row, up to the longest.
PAUSE_FIRST_SECONDS = 30.0
PAUSE_MAX_SECONDS = 600.0
# Bound the exponent before evaluating it: a long refusal history must not
# overflow and prevent the proxy from recording its next pause.
PAUSE_MAX_DOUBLINGS = math.ceil(math.log2(PAUSE_MAX_SECONDS / PAUSE_FIRST_SECONDS))
# A retry-after header above this is read as this, so that a malformed one cannot stop
# shadow mode for good.
RETRY_AFTER_MAX_SECONDS = 3600.0
# The skip reason of a shadow run in which a provider refused a call (429 or 529) or a
# call was not sent because shadow mode had paused. Its row keeps what was spent; the
# report leaves it out of the projection and counts it as ``rate_limited_runs``.
RATE_LIMITED_REASON = "rate_limited: a provider refused a shadow call (429 or 529), so the run is left out"


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


class ShadowPaused(Exception):
    """A shadow call not sent, because shadow mode had paused on its provider. Nothing
    was spent on it."""

    cost_usd = 0.0

    def __init__(self) -> None:
        super().__init__("shadow mode is paused on this provider")


def _blocks(provider: str, payload: Any) -> tuple[list, list, Any]:
    """The text parts, tool calls and stop reason of a chat reply, however odd its shape."""
    try:
        if provider == "openai":
            choice = payload["choices"][0]
            message = choice["message"]
            if message.get("refusal"):
                # A refusal may accompany text or tools and a normal stop reason.
                return [], [], "refusal"
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
    stored. A cost that is not a usable amount is stored as unknown, except that a
    provider call refused for its rate limit or load, or never sent because shadow
    mode had paused, cost nothing (``refused``)."""
    entries = [*outcome["calls"], *({**run, "model": f"check:{run['check']}"} for run in outcome.get("checks", []))]
    costed = [{**entry, "cost_usd": known_cost(entry.get("cost_usd"))} for entry in entries]
    return [{**entry, "cost_usd": known_cost(entry.get("known_cost_usd")) or 0.0}
            if entry["cost_usd"] is None and refused(entry) else entry for entry in costed]


def refused(entry: dict) -> bool:
    """Whether a stored call, or the judge step that ended a check, was refused by the
    provider for its rate limit or load (429, 529) or not sent because shadow mode had
    paused. A webhook is the customer's own service, not a provider call: its 429 is a
    failure like any other."""
    if "check" in entry:
        steps = entry.get("steps")
        last = steps[-1] if isinstance(steps, list) and steps and isinstance(steps[-1], dict) else {}
        return last.get("type") == "judge" and (last.get("status") in LIMIT_STATUSES
                                                or last.get("error") == "judge ShadowPaused")
    return entry.get("status") in LIMIT_STATUSES or entry.get("error") == "ShadowPaused"


def most_calls(policy: dict, checks: dict) -> int:
    """The most provider calls one shadow run of ``policy`` can make: each stage model
    once, and each judge step of each stage's named check. Webhooks are not counted."""
    models = {model for stage in policy["stages"] for model in stage["models"]}
    judges = sum(step.get("type") == "judge" for stage in policy["stages"] for step in checks.get(stage["check"], []))
    return len(models) + judges


def retry_after_seconds(value: Any) -> float | None:
    """A provider's ``retry-after`` header in seconds, or None when it is absent or not a
    number of seconds (an HTTP date is not read)."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return min(seconds, RETRY_AFTER_MAX_SECONDS) if math.isfinite(seconds) and seconds >= 0 else None


class ShadowLimits:
    """When shadow mode holds back so that it never competes with real traffic.

    Rate limits belong to the provider organisation, so shadow calls and the customer's
    real calls draw on the same ones. Two things are kept here, in memory:

    - A pause per provider. A 429 or 529 from the provider, on a shadow call or on real
      traffic, pauses shadow mode on that provider for every tenant: the proxy calls
      each provider with one key. The pause lasts as long as the reply's
      ``retry-after`` asks (seconds, at least one), or else 30 seconds, doubled for each
      pause in a row up to 10 minutes. A shadow run that ends with no call refused
      resets the doubling. Nothing is retried.
    - With ``max_calls_per_minute``, a cap per tenant on shadow provider calls started in
      any 60 seconds. A run reserves the most calls it can make when it starts and gives
      back the ones it did not make when it ends.

    ``clock`` gives the time in seconds since the epoch; tests pass their own.
    """

    def __init__(self, max_calls_per_minute: int | None = None, clock=time.time) -> None:
        self.max_calls_per_minute = max_calls_per_minute
        self._clock = clock
        self._pauses: dict[str, dict] = {}
        self._calls: dict[str, deque] = {}

    def limited(self, provider: str, status: int, retry_after: Any, where: str) -> bool:
        """Pause shadow mode on ``provider`` after it refused a call with ``status``.
        ``where`` says which call, for the report. True when a new pause starts; a
        refusal during a pause only lengthens it, when its retry-after asks for longer."""
        now = self._clock()
        pause = self._pauses.get(provider, {"until": now, "in_a_row": 0})
        ongoing = pause["until"] > now
        in_a_row = pause["in_a_row"] if ongoing else pause["in_a_row"] + 1
        asked = retry_after_seconds(retry_after)
        wait = asked if asked is not None else min(
            PAUSE_FIRST_SECONDS * 2 ** min(in_a_row - 1, PAUSE_MAX_DOUBLINGS), PAUSE_MAX_SECONDS)
        until = now + max(wait, 1.0)
        if ongoing and until <= pause["until"]:
            return False
        how = (f"the reply asked to wait {asked:g} seconds" if asked is not None
               else f"no retry-after, so {wait:g} seconds (pause {in_a_row} in a row)")
        self._pauses[provider] = {"until": until, "in_a_row": in_a_row,
                                  "reason": f"{where} got HTTP {status} from {provider}; {how}"}
        return not ongoing

    def paused(self, provider: str) -> dict | None:
        """The pause in force on ``provider``, or None."""
        pause = self._pauses.get(provider)
        return pause if pause is not None and pause["until"] > self._clock() else None

    def succeeded(self, provider: str) -> None:
        """A shadow run on ``provider`` ended with no call refused: the next pause starts
        again at the shortest. A pause in force is left as it is."""
        pause = self._pauses.get(provider)
        if pause is not None and self.paused(provider) is None:
            pause["in_a_row"] = 0

    def _window(self, tenant: str) -> deque:
        window = self._calls.setdefault(tenant, deque())
        while window and window[0][0] <= self._clock() - 60:
            window.popleft()
        return window

    def reserve(self, tenant: str, calls: int) -> list | None:
        """Hold room for a run of at most ``calls`` provider calls under the tenant's cap.
        None when the cap has no room; otherwise a reservation for ``release``."""
        if self.max_calls_per_minute is None:
            return [self._clock(), 0]
        window = self._window(tenant)
        if sum(held for _, held in window) + calls > self.max_calls_per_minute:
            return None
        reservation = [self._clock(), calls]
        window.append(reservation)
        return reservation

    def release(self, reservation: list, made: int) -> None:
        """The run ended having made ``made`` calls: give back the rest of its room."""
        reservation[1] = min(reservation[1], made)

    def report(self, tenant: str, providers) -> dict:
        """The pause on each of ``providers`` and the tenant's call cap, for ``GET /v1/shadow``."""
        now = self._clock()
        pauses = {}
        for provider in sorted(providers):
            pause = self.paused(provider)
            pauses[provider] = None if pause is None else {
                "paused_until": datetime.fromtimestamp(pause["until"], timezone.utc).isoformat(),
                "seconds_left": round(pause["until"] - now, 3), "reason": pause["reason"]}
        capped = self.max_calls_per_minute is not None
        return {"shadow_paused": pauses, "shadow_max_calls_per_minute": self.max_calls_per_minute,
                "shadow_calls_last_minute": sum(held for _, held in self._window(tenant)) if capped else None}


def _known_part(entry: dict) -> float:
    """What is known of an entry's cost: all of it, the known steps of a check whose
    total is unknown, or nothing."""
    cost = entry.get("cost_usd")
    return cost if cost is not None else (known_cost(entry.get("known_cost_usd")) or 0.0)


def _columns(ledger) -> set[str]:
    return {row["name"] for row in ledger._db.execute("PRAGMA table_info(shadow_runs)")}


def policy_row_digest(policy: dict) -> str:
    """What a shadow row keeps of the policy it was run with (``policy_sha256``): the
    SHA-256 of the whole policy as loaded, provenance included. It is not the freeze
    digest (``tine.policy.policy_digest``), which leaves its own field out."""
    return hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def policy_chain_digest(policy: dict) -> str:
    """What a shadow row keeps of how its policy runs (``policy_chain_sha256``): the
    SHA-256 of the frontier, the stages, the answer pattern and the answer comparison.
    Two policy files with the same digest accept and compare the same answers, whatever
    their limit, evidence or date, so the drift check of one may count the requests
    run with the other: freezing a policy again does not start its check from nothing."""
    chain = {"frontier": policy.get("frontier"), "stages": policy.get("stages"),
             "answer_pattern": policy.get("answer_pattern"), "answer_compare": answer_mode(policy)}
    return hashlib.sha256(json.dumps(chain, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def run_with(policy: dict):
    """A test for shadow rows: whether a row was run with ``policy``'s file, or with a
    file that runs the same way (``policy_chain_digest``). Rows stored before the
    second digest was kept count only for the very file."""
    file, chain = policy_row_digest(policy), policy_chain_digest(policy)
    return lambda row: (row["policy_sha256"] == file
                        or (row["policy_chain_sha256"] is not None and row["policy_chain_sha256"] == chain))


def chain_column(ledger) -> str:
    """The chain digest column for a SELECT; NULL for a ledger opened read-only that
    predates it."""
    return "policy_chain_sha256" if "policy_chain_sha256" in _columns(ledger) else "NULL AS policy_chain_sha256"


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
                    skip_reason TEXT,
                    policy_chain_sha256 TEXT
                )"""
            )
            ledger._db.execute("CREATE INDEX IF NOT EXISTS shadow_tenant ON shadow_runs(tenant)")
            for column in ("skip_reason", "policy_chain_sha256"):
                if column not in _columns(ledger):
                    ledger._db.execute(f"ALTER TABLE shadow_runs ADD COLUMN {column} TEXT")
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
        sent to the frontier without trying its stages; ``outcome`` then has no calls.
        The accepted answer is compared with the frontier's the way the policy's
        agreement compares answers (``answer_compare``)."""
        matches = None
        if outcome["accepted"] and outcome.get("comparable", True):
            matches = int(frontier_answer is not None
                          and same_answer([outcome["answer"], frontier_answer], answer_mode(policy)))
        spent = shadow_entries(outcome)
        digest = policy_row_digest(policy)
        with self._ledger._lock, self._ledger._db:
            self._ledger._db.execute(
                "INSERT INTO shadow_runs (request_id, tenant, provider, created_at, policy_sha256, accepted, stage, "
                "served_by, matches_frontier, calls_json, shadow_cost_usd, frontier_cost_usd, skip_reason, "
                "policy_chain_sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id, tenant, provider, datetime.now(timezone.utc).isoformat(), digest,
                 int(outcome["accepted"]), outcome["stage"], outcome["served_by"], matches,
                 json.dumps(spent, allow_nan=False),
                 _known_sum(entry["cost_usd"] for entry in spent), frontier_cost_usd, skip_reason,
                 policy_chain_digest(policy)))

    def adopt(self, tenant: str, policy: dict) -> int:
        """Mark the tenant's rows that were run with this very policy file, and stored
        before rows kept how their policy runs, with its chain digest
        (``policy_chain_digest``). The proxy calls this for each shadow policy when it
        starts, so that a policy frozen again later still counts them in its drift
        check. Returns the number of rows marked."""
        with self._ledger._lock, self._ledger._db:
            return self._ledger._db.execute(
                "UPDATE shadow_runs SET policy_chain_sha256=? WHERE tenant=? AND policy_sha256=? "
                "AND policy_chain_sha256 IS NULL",
                (policy_chain_digest(policy), tenant, policy_row_digest(policy))).rowcount

    def count(self, tenant: str, name: str) -> None:
        """Count one sampled request that has no row of its own: it was skipped
        (``skipped_busy``, ``skipped_stopped``, ``skipped_rate_limited`` while shadow
        mode was paused on a provider, ``skipped_rate_cap`` over the per-minute call
        cap), its run failed before its row was written (``failed``), or it was
        streamed (``streamed``)."""
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
               min_compared: int = DRIFT_MIN_COMPARED, *, policies: dict[str, dict] | None = None) -> dict[str, Any]:
        """The tenant's shadow figures, with the drift check (``drift``).

        ``policies`` maps a label to a policy being shadowed: the proxy passes the
        tenant's shadow policies by provider. Each one that has stages is checked on
        the requests that were run with that file or with one that runs the same way
        (``run_with``), against its own quality limit and its own drift baseline. The
        ``drift_*`` fields at the top describe one of them, a policy with a warning
        first, then one with a notice, and otherwise the one with the most requests;
        ``drift_policies`` holds every one when there are several. Requests run with
        none of the given policies are counted in ``drift_other_policy_requests`` and
        are in no check. ``accepted_differing_from_frontier`` and its share cover
        every request, like the figures beside them; the counts the check itself used
        are ``drift_differing``, ``drift_compared`` and ``drift_requests``.

        Without ``policies``, ``epsilon`` is the quality limit to check against: all
        the tenant's requests are then checked together, whatever policy they were run
        with, at the default allowance. With neither, nothing is checked."""
        with self._ledger._lock:
            # A ledger opened read-only may predate the column: nothing in it was skipped.
            skipped = "skip_reason" if "skip_reason" in _columns(self._ledger) else "NULL AS skip_reason"
            # In the order the runs were stored: the drift check weighs the latest ones apart.
            rows = self._ledger._db.execute(
                "SELECT accepted,matches_frontier,shadow_cost_usd,frontier_cost_usd,calls_json,policy_sha256,"
                f"{skipped},{chain_column(self._ledger)} FROM shadow_runs WHERE tenant=? ORDER BY created_at, rowid",
                (tenant,)).fetchall()
        # What every run spent counts, but a run in which a provider refused a call does
        # not show what the policy would do, so it is left out of everything else.
        calls = [entry for row in rows for entry in json.loads(row["calls_json"])]
        limited = sum(row["skip_reason"] == RATE_LIMITED_REASON for row in rows)
        rows = [row for row in rows if row["skip_reason"] != RATE_LIMITED_REASON]
        accepted = [row for row in rows if row["accepted"]]
        compared = [row for row in accepted if row["matches_frontier"] is not None]
        matching = sum(row["matches_frontier"] == 1 for row in compared)
        known = [row for row in rows if row["shadow_cost_usd"] is not None and row["frontier_cost_usd"] is not None]
        frontier_cost = math.fsum(row["frontier_cost_usd"] for row in known)
        # What the policy would have cost: every cheaper call it made, plus the
        # frontier where no cheaper answer was accepted.
        policy_cost = math.fsum(row["shadow_cost_usd"] + (0.0 if row["accepted"] else row["frontier_cost_usd"])
                                for row in known)
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
            # Which step of each named check rejected: a step that failed counts, since a
            # check fails closed. Runs stored before steps were kept are not counted.
            "check_step_rejections": step_rejections(calls),
            **coverage(len(rows), self.counts(tenant), rate_limited_runs=limited),
            **_drift_fields(rows, epsilon, min_compared, policies),
            # Over every request, like the matching count above; the drift check's own
            # counts are drift_differing and drift_compared.
            "accepted_differing_from_frontier": len(compared) - matching,
            "accepted_differing_from_frontier_share": (len(compared) - matching) / len(compared) if compared else None,
            "scope": SHADOW_SCOPE,
        }


NOT_COMPARED = ("not checked: this policy's answers are accepted by a named check and are never compared with the "
                "frontier's")


def _drift_fields(rows, epsilon: float | None, min_compared: int, policies: dict[str, dict] | None) -> dict:
    """The drift fields of a shadow report, from a tenant's rows in stored order
    (``ShadowStore.report``)."""
    if policies is None:
        return {**_drift_of(rows, epsilon, min_compared),
                "drift_policy": None, "drift_policies": None, "drift_other_policy_requests": None}
    mine = {label: run_with(policy) for label, policy in policies.items()}
    checks = {label: _drift_of([row for row in rows if mine[label](row)], _quality_limit(policy), min_compared, policy)
              for label, policy in policies.items() if policy.get("stages")}
    other = sum(not any(test(row) for test in mine.values()) for row in rows)
    if not checks:  # no policy with stages, so no limit to check against
        return {**_drift_of(rows, None, min_compared),
                "drift_policy": None, "drift_policies": None, "drift_other_policy_requests": other}
    # A policy with a warning is shown first, then one with a notice, else the one with
    # the most requests.
    shown = max(checks, key=lambda label: (checks[label]["drift_check"] == "warning",
                                           checks[label]["drift_check"] == "notice", checks[label]["drift_requests"]))
    return {**checks[shown], "drift_policy": shown, "drift_policies": checks if len(checks) > 1 else None,
            "drift_other_policy_requests": other}


def _quality_limit(policy: dict) -> float | None:
    """A policy's epsilon, or None when the file states no usable one."""
    guarantee = policy.get("guarantee")
    epsilon = guarantee.get("epsilon") if isinstance(guarantee, dict) else None
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        return None
    return float(epsilon) if 0 < epsilon < 1 else None


def _drift_of(rows, epsilon: float | None, min_compared: int, policy: dict | None = None) -> dict:
    """``drift`` on shadow rows in stored order: the requests the policy's stages were
    run on. A request sent to the frontier untried (it carries tools) is not of the
    kind the policy was certified on and is left out. An answer accepted by a named
    check stays in as a request and is not compared."""
    flags = [(bool(row["accepted"]) and row["matches_frontier"] == 0,
              bool(row["accepted"]) and row["matches_frontier"] is not None)
             for row in rows if row["skip_reason"] is None]
    if policy is not None and all(stage["check"] not in BUILT_IN_CHECKS for stage in policy["stages"]):
        # Nothing this policy accepts is ever compared, so there is nothing to check.
        return {**drift(0, 0, None, min_compared, requests=len(flags)), "drift_epsilon": epsilon,
                "drift_check": NOT_COMPARED}
    return drift(sum(compared for _, compared in flags), sum(differs for differs, _ in flags), epsilon, min_compared,
                 requests=len(flags), baseline=policy.get("drift_baseline") if policy else None,
                 latest=drift_scopes(flags))


def step_rejections(entries) -> dict[str, dict[str, int]]:
    """Check name to step type to the number of times a step of that type rejected an
    answer, from stored check entries that list their ``steps``."""
    counts: dict[str, Counter] = {}
    for entry in entries:
        steps = entry.get("steps")
        if not isinstance(entry.get("check"), str) or not isinstance(steps, list):
            continue
        for step in steps:
            if isinstance(step, dict) and step.get("passed") is False:
                counts.setdefault(entry["check"], Counter())[str(step.get("type"))] += 1
    return {name: dict(sorted(counts[name].items())) for name in sorted(counts)}


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


def coverage(shadowed: int, counts: dict[str, int] | None, rate_limited_runs: int = 0) -> dict[str, Any]:
    """What became of the sampled requests: how many were shadowed, and why the rest
    were not.

    ``sampled_requests`` counts the sampled requests a shadow run could have taken
    (answered by the frontier without streaming) once their run ended or was skipped:
    the ``shadowed`` ones, which have a row, the skipped and failed ones, which are
    counted, and the ``rate_limited_runs``, whose rows are left out of the projection.
    Requests skipped while shadow mode was paused on a provider are
    ``skipped_rate_limited``; those over the per-minute call cap are
    ``skipped_rate_cap``. ``shadow_coverage`` is the share that were shadowed. Busy
    periods lose more of their sample, so a coverage below one means the projection
    leans to quieter ones. Streamed requests are counted apart: they were sampled,
    but shadow mode cannot run them. The counts start when a ledger is first opened
    by a proxy that keeps them; all are None for a ledger that has none to read.
    """
    if counts is None:
        return dict.fromkeys(("sampled_requests", "skipped_busy", "skipped_stopped", "failed_shadow_runs",
                              "skipped_rate_limited", "skipped_rate_cap", "rate_limited_runs",
                              "streamed_not_shadowed", "shadow_coverage"))
    sampled = (shadowed + counts["skipped_busy"] + counts["skipped_stopped"] + counts["failed"]
               + counts["skipped_rate_limited"] + counts["skipped_rate_cap"] + rate_limited_runs)
    return {"sampled_requests": sampled, "skipped_busy": counts["skipped_busy"],
            "skipped_stopped": counts["skipped_stopped"], "failed_shadow_runs": counts["failed"],
            "skipped_rate_limited": counts["skipped_rate_limited"], "skipped_rate_cap": counts["skipped_rate_cap"],
            "rate_limited_runs": rate_limited_runs, "streamed_not_shadowed": counts["streamed"],
            "shadow_coverage": shadowed / sampled if sampled else None}


def drift_allowance(epsilon: float, baseline: dict | None = None) -> dict:
    """The shares of differing answers the drift check allows for, and what they rest on.

    ``baseline`` is a policy's ``drift_baseline`` block (``tine.policy.freeze_policy``):
    of ``tasks`` graded tasks the policy was certified and tested on, ``compared`` had
    an accepted answer that was compared with the frontier's, ``differing`` of those
    differed and ``differing_not_lost`` differed without being a loss.

    - ``drift_allowance``, a share of requests: ``epsilon`` plus the exact
      (Clopper-Pearson) upper bound on ``differing_not_lost / tasks`` at
      ``DRIFT_BASELINE_CONFIDENCE``.
    - ``drift_allowance_accepted``, a share of accepted answers: the same two parts
      per compared answer, ``epsilon * tasks / compared`` plus the upper bound on
      ``differing_not_lost / compared``. It does not shrink when requests of another
      kind, which the policy never accepts, are mixed into the traffic.
    - ``drift_history``: the upper bounds on ``differing / tasks`` and ``differing /
      compared``, the shares that differed on the graded tasks, losses included. A
      share clearly above them is a change from the graded tasks, whether or not it is
      above the allowance.

    All of it holds only if the graded tasks are a sample of the traffic. Without a
    baseline the allowance is ``epsilon`` plus ``DRIFT_DEFAULT_ALLOWANCE`` of requests
    and ``DRIFT_DEFAULT_ALLOWANCE_ACCEPTED`` of accepted answers, rough levels that
    show only a large change.
    """
    none = {"drift_allowance_accepted": None, "drift_epsilon_accepted": None, "drift_history": None,
            "drift_accepted_share": None}
    if not isinstance(baseline, dict):
        return {"drift_basis": "default", "drift_allowance": min(1.0, epsilon + DRIFT_DEFAULT_ALLOWANCE),
                "drift_baseline": None, **none, "drift_allowance_accepted": DRIFT_DEFAULT_ALLOWANCE_ACCEPTED,
                "drift_epsilon_accepted": epsilon}
    # Imported here, not at the top: it loads numpy and scipy, which the proxy must start
    # without. They are loaded the first time a drift check is worked out.
    from tine.calibration import clopper_pearson_upper

    alpha = 1 - DRIFT_BASELINE_CONFIDENCE
    tasks, harmless = baseline["tasks"], baseline["differing_not_lost"]
    compared, differing = baseline.get("compared"), baseline.get("differing")
    among = type(compared) is int and harmless <= compared <= tasks and compared > 0
    recorded = {"tasks": tasks, "differing_not_lost": harmless, "upper": clopper_pearson_upper(harmless, tasks, alpha)}
    levels = {"drift_basis": "baseline", "drift_allowance": min(1.0, epsilon + recorded["upper"]),
              "drift_baseline": recorded, **{**none, "drift_accepted_share": compared / tasks if among else None}}
    if among:
        recorded.update(compared=compared, upper_accepted=clopper_pearson_upper(harmless, compared, alpha))
        levels["drift_epsilon_accepted"] = epsilon * tasks / compared
        levels["drift_allowance_accepted"] = min(1.0, levels["drift_epsilon_accepted"] + recorded["upper_accepted"])
    if type(differing) is int and harmless <= differing <= (compared if among else tasks):
        levels["drift_history"] = {
            "tasks": tasks, "differing": differing, "upper": clopper_pearson_upper(differing, tasks, alpha),
            **({"compared": compared, "upper_accepted": clopper_pearson_upper(differing, compared, alpha)}
               if among else {})}
    return levels


def drift_scopes(flags) -> list[dict]:
    """The counts of the latest drift requests, for ``drift``'s ``latest``.

    ``flags`` holds, per drift request in stored order, whether its accepted answer
    differs from the frontier's and whether an accepted answer was compared. Entry
    ``k - 1`` of the result counts the requests after the first ``DRIFT_BLOCK * k``.
    """
    flags = list(flags)
    # Running totals, so that a long history is read once and not once per block.
    differing = [0, *accumulate(int(bool(differs)) for differs, _ in flags)]
    compared = [0, *accumulate(int(bool(compared)) for _, compared in flags)]
    return [{"requests": len(flags) - start, "compared": compared[-1] - compared[start],
             "differing": differing[-1] - differing[start]} for start in range(DRIFT_BLOCK, len(flags), DRIFT_BLOCK)]


def _log_w(differing, trials, allowance: float, upper: float):
    """Log of the likelihood of ``differing`` in ``trials`` averaged over differing
    rates spread evenly between ``allowance`` and ``upper``, over its likelihood at
    ``allowance``. The average is a difference of two incomplete beta integrals; it
    is taken on the tails away from the mass, to keep the difference accurate. The
    counts may be arrays: one value per scope."""
    import numpy as np
    from scipy import special

    differing, trials = np.asarray(differing, dtype=float), np.asarray(trials, dtype=float)
    a, b = differing + 1.0, trials - differing + 1.0
    # Where the mass sits right of the interval the lower tails are the small ones.
    lower = special.betainc(a, b, upper) - special.betainc(a, b, allowance)
    tails = special.betaincc(a, b, allowance) - (special.betaincc(a, b, upper) if upper < 1.0 else 0.0)
    mass = np.where(a / (trials + 2.0) > upper, lower, tails)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mass = np.where(mass > 0.0, np.log(np.where(mass > 0.0, mass, 1.0)), -np.inf)
    return (special.betaln(a, b) + log_mass - math.log(upper - allowance)
            - differing * math.log(allowance) - (trials - differing) * math.log1p(-allowance))


def _log_m(differing, trials, allowance: float, epsilon: float):
    """Log of the evidence one scope gives: half from differing rates anywhere above
    the allowance, half from rates within twice the quality limit above it, where a
    change is hardest to see. With no trial it is 1 (log 0)."""
    import numpy as np

    if allowance >= 1.0:  # nothing can be above it
        return np.full(np.shape(differing), -np.inf)
    wide = _log_w(differing, trials, allowance, 1.0)
    near = _log_w(differing, trials, allowance, min(1.0, allowance + 2 * epsilon))
    return np.logaddexp(wide, near) - math.log(2)


def drift_evidence(scopes, allowance: float, epsilon: float, min_compared: int = DRIFT_MIN_COMPARED, *,
                   accepted_allowance: float | None = None,
                   accepted_epsilon: float | None = None) -> tuple[float, dict | None]:
    """How strongly the counts speak against "each request's accepted answer differs
    with probability at most ``allowance``", and the scope that says most.

    ``scopes[0]`` counts all the drift requests and ``scopes[k]`` those after the
    first ``DRIFT_BLOCK * k`` (``requests``, ``compared``, ``differing``). Each scope
    gives a likelihood ratio (``_log_m``); the evidence is their weighted sum, half
    for all the requests and ``1 / (2k(k+1))`` for scope ``k``, weights that add up
    to one. A scope counts only once it holds ``min_compared`` compared answers.

    With ``accepted_allowance`` the same is done a second time among the compared
    answers alone, against "each compared answer differs with probability at most
    ``accepted_allowance``" (``accepted_epsilon`` is the quality limit as a share of
    accepted answers), and the evidence is the mean of the two sums. Requests of a
    kind the policy never accepts thin out the share of requests and leave the share
    of accepted answers as it is.

    If both statements hold given the requests before, each sum is a nonnegative
    supermartingale that starts at or below one and so is their mean, so the chance
    that it ever reaches ``DRIFT_EVIDENCE_NEEDED`` is at most one over that (Ville's
    inequality), however often the report is read.
    """
    import numpy as np

    scopes = list(scopes)
    requests, compared, differing = (np.array([scope[name] for scope in scopes], dtype=float)
                                     for name in ("requests", "compared", "differing"))
    weights = np.array([0.5 if k == 0 else 0.5 / (k * (k + 1)) for k in range(len(scopes))])
    kinds = [("requests", requests, allowance, epsilon)]
    if accepted_allowance is not None:
        kinds.append(("accepted", compared, accepted_allowance, accepted_epsilon))
    total, largest = 0.0, None
    for among, trials, level, step in kinds:
        terms = np.where(compared >= min_compared,
                         weights / len(kinds) * np.exp(np.minimum(_log_m(differing, trials, level, step), 700.0)), 0.0)
        total += float(terms.sum())
        k = int(terms.argmax()) if len(terms) else 0
        if len(terms) and compared[k] >= min_compared and (largest is None or terms[k] > largest[0]):
            largest = (float(terms[k]), {**{name: int(scopes[k][name]) for name in ("requests", "compared", "differing")},
                                         "latest": k > 0, "among": among})
    return total, largest[1] if largest else None


def _counted(scope: dict, cheaper: str = "") -> str:
    """The counts of one scope in words: among all its requests or its accepted answers."""
    where = f"the {'latest ' if scope['latest'] else ''}{scope['requests']:,} shadowed requests the policy was run on"
    start = f"{scope['differing']:,} of {scope['compared']:,} accepted {cheaper}answers differ from "
    whose = "the frontier model's answer" if cheaper else "the frontier's"
    if scope["among"] == "accepted":
        return f"{start}{whose}, {scope['differing'] / scope['compared']:.2%} of them, in {where}"
    return f"{start}{whose}, {scope['differing'] / scope['requests']:.2%} of {where}"


def _allowed(levels: dict, epsilon: float, among: str) -> str:
    """What the certification allows for, in words, as a share of requests or of
    accepted answers."""
    recorded = levels["drift_baseline"]
    if among == "accepted":
        if recorded:
            return (f"the {levels['drift_allowance_accepted']:.2%} of accepted answers the certification allows for: "
                    f"the quality limit of {epsilon:.2%} of requests, which is {levels['drift_epsilon_accepted']:.2%} "
                    f"of accepted answers when {levels['drift_accepted_share']:.2%} of requests are accepted as on "
                    f"the graded tasks, plus {recorded['upper_accepted']:.2%} for answers that differed without a "
                    f"loss among the {recorded['compared']:,} compared on the graded tasks the policy was certified "
                    "and tested on")
        return (f"{levels['drift_allowance_accepted']:.2%} of accepted answers, a default used because this policy "
                "file records no baseline of its own")
    if recorded:
        return (f"the {levels['drift_allowance']:.2%} the certification allows for: the quality limit of "
                f"{epsilon:.2%} plus {recorded['upper']:.2%} for answers that differed without a loss on the "
                f"{recorded['tasks']:,} graded tasks the policy was certified and tested on")
    return (f"{levels['drift_allowance']:.2%}: the quality limit of {epsilon:.2%} plus a default "
            f"{DRIFT_DEFAULT_ALLOWANCE:.2%}, used because this policy file records no baseline of its own")


def drift(compared: int, differing: int, epsilon: float | None, min_compared: int = DRIFT_MIN_COMPARED,
          *, requests: int | None = None, baseline: dict | None = None, latest=()) -> dict:
    """Whether accepted answers differ from the frontier's clearly more often than the
    policy's certification allows for, so that this traffic cannot confirm it.

    The proxy cannot see right or wrong, only whether two answers are the same. The
    policy is certified to lose at most ``epsilon`` of all tasks against the frontier.
    An accepted answer equal to the frontier's loses nothing against it, and neither
    does a request the policy sends to the frontier. So the differing answers are
    counted as a share of ``requests``: the shadowed requests the policy's stages
    were run on (``compared`` when not given).

    A differing answer is not a loss. The frontier can be the one that is wrong, both
    can be wrong in different ways, and the same value can be written in two ways. A
    healthy policy therefore differs more often than it loses, and the share is not
    compared with ``epsilon`` alone but with ``drift_allowance``: ``epsilon`` plus the
    share that differed without a loss when the policy was frozen (``baseline``, the
    policy's ``drift_baseline``), or plus a default for a policy file without one.

    Traffic is not always of the certified kind alone. Requests of another kind that
    the policy tries and never accepts make the share of requests small whatever the
    accepted answers do. So the share among the compared answers is checked as well,
    against the same allowance per accepted answer when the baseline says what
    share of requests the policy accepted on its graded tasks, or against a default
    for a policy file without one (``drift_allowance``).

    The check warns only when a share is clearly above its allowance: when
    ``drift_evidence`` reaches ``DRIFT_EVIDENCE_NEEDED``, on at least ``min_compared``
    compared answers. ``latest`` holds the counts of the latest requests
    (``drift_scopes``), which are weighed apart so that a change late in a history
    shows; without it only all the requests together are weighed. The verdict is
    worked out afresh at each read: a warning goes away when later requests no
    longer support it.

    Short of a warning, a policy with a baseline gets a notice (``drift_notice``) when
    a share is clearly above the share that differed on the graded tasks, losses
    included: the traffic is then unlike the graded tasks, and the check cannot tell
    a loss that rose within the limit, or beyond it by less than the check can see,
    from harmless differences that became more frequent.

    What the check cannot see: a loss just above the limit; a loss up to about twice
    the limit at the size of a pilot, or at any size where harmless differences are
    common; a cheaper answer equal to a wrong frontier answer. If harmless differences
    become more frequent than they were at freezing, it can warn with the loss under
    the limit. Cheaper models that agree on wrong answers push the share up while they
    push the projected saving up too, and that is the case it is for. A warning or a
    notice asks for grading; neither is a measured loss.
    """
    if epsilon is not None and (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
                                or not 0 < epsilon < 1):
        raise ValueError("epsilon must be a number strictly between 0 and 1, or None")
    requests = compared if requests is None else requests
    share = differing / requests if requests else None
    result = {"accepted_differing_from_frontier": differing,
              "accepted_differing_from_frontier_share": differing / compared if compared else None,
              "drift_differing": differing, "drift_compared": compared,
              "drift_requests": requests, "drift_share": share,
              "drift_epsilon": epsilon, "drift_min_compared": min_compared, "drift_warning": None,
              "drift_notice": None, "drift_basis": None, "drift_allowance": None, "drift_baseline": None,
              "drift_allowance_accepted": None, "drift_epsilon_accepted": None, "drift_accepted_share": None,
              "drift_history": None, "drift_evidence": None, "drift_notice_evidence": None,
              "drift_evidence_needed": DRIFT_EVIDENCE_NEEDED, "drift_scope": None}
    if epsilon is None:
        result["drift_check"] = "not checked: the policy's quality limit was not given"
        return result
    levels = drift_allowance(epsilon, baseline)
    result.update(levels)
    if compared < min_compared or not requests:
        result["drift_check"] = (f"not checked yet: {compared:,} accepted answers compared with the frontier's, "
                                 f"{min_compared:,} needed")
        return result
    scopes = [{"requests": requests, "compared": compared, "differing": differing}, *latest]
    allowance, among, step = (levels[name] for name in ("drift_allowance", "drift_allowance_accepted",
                                                         "drift_epsilon_accepted"))
    evidence, scope = drift_evidence(scopes, allowance, epsilon, min_compared,
                                     accepted_allowance=among, accepted_epsilon=step)
    result["drift_evidence"], result["drift_scope"] = min(evidence, DRIFT_EVIDENCE_CAP), scope
    grade = "A different answer is not always a wrong one; grade a sample of these requests"
    if evidence >= DRIFT_EVIDENCE_NEEDED:
        # The scope that says most: all the requests or the latest ones, as a share of
        # requests or of accepted answers.
        result["drift_check"] = "warning"
        result["drift_warning"] = (
            f"Cannot confirm the certification on this traffic: {_counted(scope, 'cheaper ')}. That is clearly more "
            f"than {_allowed(levels, epsilon, scope['among'])}. {grade} before relying on the policy or its "
            "projected saving.")
        return result
    whole = {**scopes[0], "latest": False, "among": "requests"}
    quiet = f"{_counted(whole)}. That is not clearly above {_allowed(levels, epsilon, 'requests')}"
    if among is not None:
        quiet += f". Of the accepted answers alone {differing / compared:.2%} differ, against {among:.2%} allowed for"
    history = levels["drift_history"]
    if history and history["upper"] < allowance:
        # Not above what the certification allows for. Is it above the graded tasks' own share?
        seen = among is not None and history["upper_accepted"] < among
        evidence, scope = drift_evidence(scopes, history["upper"], epsilon, min_compared,
                                         accepted_allowance=history["upper_accepted"] if seen else None,
                                         accepted_epsilon=step)
        result["drift_notice_evidence"] = min(evidence, DRIFT_EVIDENCE_CAP)
        if evidence >= DRIFT_EVIDENCE_NEEDED:
            of, bound = (("compared", "upper_accepted") if scope["among"] == "accepted" else ("tasks", "upper"))
            result["drift_scope"] = scope
            result["drift_check"] = "notice"
            result["drift_notice"] = (
                f"Accepted answers differ from the frontier's more often than on the graded tasks: {_counted(scope)}. "
                f"On the graded tasks the policy was certified and tested on, {history['differing']:,} of "
                f"{history[of]:,} did, {history['differing'] / history[of]:.2%}, and at most {history[bound]:.2%} "
                f"with {DRIFT_BASELINE_CONFIDENCE:.0%} confidence. The share is not clearly above "
                f"{_allowed(levels, epsilon, scope['among'])}. The check cannot tell a higher loss from more "
                f"harmless differences. {grade} to find out which.")
            return result
    if levels["drift_baseline"]:
        wide = levels["drift_baseline"]["upper"] >= DRIFT_DEFAULT_ALLOWANCE
        result["drift_check"] = (f"no clear sign of drift: {quiet}"
                                 + (". Answers that differed without a loss were common on the graded tasks, so "
                                    "only a large change shows" if wide else ""))
    else:
        result["drift_check"] = (f"rough check, no warning: {quiet}. Only a large change shows at this level; freeze "
                                 "the policy again from recorded answers to record a baseline")
    return result
