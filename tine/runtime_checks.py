"""Named checks the proxy can run on a cheaper model's answer.

A frozen policy names the check each stage was certified with, for example
``tests+judge:haiku``. The proxy cannot guess what that means, so the operator says:
a configuration maps each name to steps, run in order, all of which must pass.

``webhook``  POST the request and the cheaper model's response to a URL the operator
             controls (their CI, a test runner beside their repository) and read back
             ``{"passed": true|false, "cost_usd": optional}``. This is how a
             customer's own tests are run: the proxy never executes model-written
             code itself.
``judge``    Ask a Claude model, through the Anthropic provider configured for the
             proxy, whether the response resolves the request, with the same
             instructions the measured judges used (``tine.judge``).

The guarantee in a policy describes the check that was measured. A configured check
must do the same thing (the same tests, the same judge model, effort and threshold)
or the guarantee does not describe what runs. In particular the measured judges read
a GitHub issue and a unified diff; here the judge reads the request's user text and
the response text, so its measured catch rate carries over only to traffic of that
shape.

The text the judge reads is not trusted: an end user, a retrieved document or the
cheaper model wrote it. Two things are done about that here. The tags that frame the
case are made inert wherever they occur inside it, so text cannot step outside its
frame. And a case too long for the judge to read whole is rejected without asking
it, because the judge's prompt keeps only the start of each text and would otherwise
rule on a request it never saw the end of. For the same reason the judge is not asked
about a reply that calls a tool or was cut off: it would rule on the words and the
proxy would serve an action, or half an answer, that it never read. Each rejection
carries a reason (``request_too_long``, ``not_a_finished_reply``,
``response_too_long``). A request too long is known before any model answers it, so
a check with a judge step says so up front (``skip_reason``) and the stage is
skipped without calling its models. The instructions themselves are the measured
ones and do not tell the judge to distrust the case; how a real judge reacts to
text that argues with it has not been measured.

A webhook token is read from the environment variable a step names in
``bearer_env``. The proxy's own settings and the provider keys live in the same
environment, so a step may not name a ``TINE_`` variable or a provider key
variable, and a token is sent only over https. Check files are operator
configuration: whoever can edit them chooses where requests and tokens go.

Every step fails closed: an error, a timeout, a malformed reply or a refusal is a
rejection, so the request moves on to the next model. What a step costs is returned
and becomes routing cost. A cost a webhook reports that is not a usable amount is
unknown, never part of a sum.
"""

from __future__ import annotations

import math
import os
import re
from urllib.parse import urlsplit

from tine.judge import MAX_ISSUE_CHARS, MAX_PATCH_CHARS, build_messages, parse_verdict
from tine.policy import describe_failure, known_cost
from tine.shadow import plain_finish, response_text

STEP_TYPES = ("webhook", "judge")
EFFORTS = ("low", "medium", "high")
# A check holds a caller's request open (live routing) or a shadow slot. Ten minutes
# is already long for either; a longer wait is refused at startup.
MAX_WEBHOOK_TIMEOUT_SECONDS = 600
# Variables a webhook step may not send as its token: the proxy's own settings (tenant
# tokens, provider keys) and the providers' usual key variables.
RESERVED_TOKEN_PREFIX = "TINE_"
RESERVED_TOKEN_VARIABLES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
_FRAME_TAG = re.compile(r"<\s*(/?)\s*(issue|patch)\s*>", flags=re.IGNORECASE)


def validate_checks(checks) -> dict:
    """Reject a check configuration this module cannot run. Returns it unchanged."""
    if not isinstance(checks, dict):
        raise ValueError("checks must map check names to lists of steps")
    for name, steps in checks.items():
        if not isinstance(name, str) or not name or name in ("agreement", "none"):
            raise ValueError("A check needs a name other than the built-in ones")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Check {name!r} needs at least one step")
        for step in steps:
            if not isinstance(step, dict) or step.get("type") not in STEP_TYPES:
                raise ValueError(f"Check {name!r}: a step's type must be one of {STEP_TYPES}")
            if step["type"] == "webhook":
                try:
                    parsed = urlsplit(step.get("url") or "")
                    parsed.port
                except ValueError:
                    parsed = None
                if (parsed is None or parsed.scheme not in ("http", "https") or not parsed.netloc
                        or parsed.username or parsed.password):
                    raise ValueError(f"Check {name!r}: a webhook needs an HTTP(S) url without credentials")
                timeout = step.get("timeout_seconds", 60)
                if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                        or not 0 < timeout <= MAX_WEBHOOK_TIMEOUT_SECONDS):
                    raise ValueError(f"Check {name!r}: timeout_seconds must be positive and at most "
                                     f"{MAX_WEBHOOK_TIMEOUT_SECONDS}")
                if "bearer_env" in step:
                    variable = step["bearer_env"]
                    if not isinstance(variable, str) or not variable:
                        raise ValueError(f"Check {name!r}: bearer_env must name an environment variable")
                    # Compared in capitals: on some systems variable names ignore case.
                    if (variable.upper().startswith(RESERVED_TOKEN_PREFIX)
                            or variable.upper() in RESERVED_TOKEN_VARIABLES):
                        raise ValueError(
                            f"Check {name!r}: bearer_env {variable!r} names a setting of the proxy or a provider "
                            f"key. A webhook token needs a variable of its own, not starting with "
                            f"{RESERVED_TOKEN_PREFIX!r} and not one of {', '.join(RESERVED_TOKEN_VARIABLES)}")
                    if parsed.scheme != "https":
                        raise ValueError(f"Check {name!r}: a webhook that is sent a token (bearer_env) needs an "
                                         "https url")
            else:
                if not isinstance(step.get("model"), str) or not step["model"]:
                    raise ValueError(f"Check {name!r}: a judge needs a model")
                if step.get("effort", "medium") not in EFFORTS:
                    raise ValueError(f"Check {name!r}: effort must be one of {EFFORTS}")
                threshold = step.get("threshold")
                if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                              or not 0 <= threshold <= 1):
                    raise ValueError(f"Check {name!r}: threshold must be between 0 and 1")
    return checks


def token_variables(steps: list) -> set[str]:
    """The environment variables the webhook steps of one check send as tokens."""
    return {step["bearer_env"] for step in steps if step.get("type") == "webhook" and step.get("bearer_env")}


def request_text(body: dict) -> str:
    """The user's side of a chat request, as plain text."""
    parts = []
    for message in body.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(block["text"] for block in content
                         if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str))
    return "\n\n".join(parts)


def inert(text: str) -> str:
    """``text`` with the judge's frame tags (``<issue>``, ``</patch>`` and so on) made
    harmless, so that text inside the frame cannot close it and continue outside."""
    return _FRAME_TAG.sub(lambda tag: f"&lt;{tag.group(1)}{tag.group(2)}&gt;", text)


def build_checks(checks: dict, *, upstream, tenant: str, provider: str, body: dict,
                 anthropic_url: str, anthropic_key: str, cost_of) -> dict:
    """Check functions for one request. ``cost_of(provider, model, payload)`` prices a
    provider response from its usage, or returns None when it cannot."""

    async def webhook(step: dict, result: dict) -> dict:
        headers = {"content-type": "application/json"}
        if step.get("bearer_env"):
            token = os.environ.get(step["bearer_env"])
            if not token:
                raise ValueError("webhook credential is not set")
            headers["authorization"] = f"Bearer {token}"
        response = await upstream.post(
            step["url"], headers=headers, timeout=step.get("timeout_seconds", 60),
            json={"tenant": tenant, "provider": provider, "model": result.get("model"),
                  "request": body, "response": result.get("payload")})
        response.raise_for_status()
        reply = response.json()
        if not isinstance(reply, dict) or not isinstance(reply.get("passed"), bool):
            raise ValueError("webhook reply needs a boolean 'passed'")
        # The tenant's CI can report anything, NaN and a 400-digit integer included.
        return {"passed": reply["passed"], "cost_usd": known_cost(reply.get("cost_usd", 0.0))}

    issue = inert(request_text(body))
    # The judge's prompt keeps the start of each text. It would not see the newest
    # turn of a long conversation or the end of a long answer, so it is not asked.
    request_too_long = len(issue) > MAX_ISSUE_CHARS

    async def judge(step: dict, result: dict, tested: bool) -> dict:
        patch = response_text(provider, result.get("payload"))
        if not patch:
            return {"passed": False, "cost_usd": 0.0}
        if request_too_long:
            return {"passed": False, "cost_usd": 0.0, "reason": "request_too_long"}
        if not plain_finish(provider, result.get("payload")):
            # The judge reads the reply's text. A reply that also calls a tool would be
            # served with an action the judge never saw, and one cut off by the token
            # limit is not the whole answer. Neither is asked about; both are rejected.
            return {"passed": False, "cost_usd": 0.0, "reason": "not_a_finished_reply"}
        patch = inert(patch)
        if len(patch) > MAX_PATCH_CHARS:
            return {"passed": False, "cost_usd": 0.0, "reason": "response_too_long"}
        messages = build_messages({"repository": step.get("repository", "the customer's"),
                                   "issue": issue, "patch": patch}, tests_passed=tested)
        response = await upstream.post(
            anthropic_url, headers={"x-api-key": anthropic_key, "anthropic-version": "2023-06-01",
                                    "content-type": "application/json"},
            json={"model": step["model"], "max_tokens": step.get("max_tokens", 8000),
                  "system": messages[0]["content"],
                  "messages": [{"role": "user", "content": messages[1]["content"]}],
                  "output_config": {"effort": step.get("effort", "medium")}})
        response.raise_for_status()
        payload = response.json()
        cost = cost_of("anthropic", step["model"], payload)
        if payload.get("stop_reason") == "refusal":
            return {"passed": False, "cost_usd": cost}
        try:
            verdict = parse_verdict(response_text("anthropic", payload) or "")
        except ValueError:
            return {"passed": False, "cost_usd": cost}
        threshold = step.get("threshold")
        passed = verdict["verdict"] == "accept" if threshold is None else verdict["probability"] >= threshold
        return {"passed": passed, "cost_usd": cost}

    def make(steps: list[dict]):
        async def check(result: dict) -> dict:
            costs, passed, tested, reason, failure = [], True, False, None, {}
            for step in steps:
                try:
                    if step["type"] == "webhook":
                        verdict = await webhook(step, result)
                    else:
                        # The judge is told the tests passed only if a test step did, just before.
                        verdict = await judge(step, result, tested)
                except Exception as error:  # fail closed: an unusable check rejects
                    verdict = {"passed": False, "cost_usd": None}
                    # Kept so that a report can say which step failed and how (an HTTP status, if any).
                    failure = describe_failure(error)
                    failure["error"] = f"{step['type']} {failure['error']}"
                costs.append(verdict["cost_usd"])
                if not verdict["passed"]:
                    passed, reason = False, verdict.get("reason")
                    break
                tested = tested or step["type"] == "webhook"
            # One step of unknown cost makes the check's cost unknown; what the other
            # steps cost is still reported, so known spending is not lost with it.
            known = math.fsum(cost for cost in costs if cost is not None)
            return {"passed": passed, "cost_usd": None if any(cost is None for cost in costs) else known,
                    "known_cost_usd": known, **({"reason": reason} if reason else {}), **failure}
        # Every step must pass, so a judge that cannot read the whole request rejects
        # whatever a model answers: the stage need not call one (tine.policy).
        check.skip_reason = ("request_too_long" if request_too_long
                             and any(step["type"] == "judge" for step in steps) else None)
        return check

    return {name: make(steps) for name, steps in checks.items()}
