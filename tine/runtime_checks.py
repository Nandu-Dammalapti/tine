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

Every step fails closed: an error, a timeout, a malformed reply or a refusal is a
rejection, so the request moves on to the next model. What a step costs is returned
and becomes routing cost.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from tine.judge import build_messages, parse_verdict
from tine.shadow import response_text

STEP_TYPES = ("webhook", "judge")
EFFORTS = ("low", "medium", "high")


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
                if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
                    raise ValueError(f"Check {name!r}: timeout_seconds must be positive")
                if "bearer_env" in step and (not isinstance(step["bearer_env"], str) or not step["bearer_env"]):
                    raise ValueError(f"Check {name!r}: bearer_env must name an environment variable")
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
        cost = reply.get("cost_usd", 0.0)
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
            cost = None
        return {"passed": reply["passed"], "cost_usd": cost}

    async def judge(step: dict, result: dict) -> dict:
        patch = response_text(provider, result.get("payload"))
        if not patch:
            return {"passed": False, "cost_usd": 0.0}
        messages = build_messages({"repository": step.get("repository", "the customer's"),
                                   "issue": request_text(body), "patch": patch})
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

    runners = {"webhook": webhook, "judge": judge}

    def make(steps: list[dict]):
        async def check(result: dict) -> dict:
            costs, passed = [], True
            for step in steps:
                try:
                    verdict = await runners[step["type"]](step, result)
                except Exception:  # fail closed: an unusable check rejects
                    verdict = {"passed": False, "cost_usd": None}
                costs.append(verdict["cost_usd"])
                if not verdict["passed"]:
                    passed = False
                    break
            return {"passed": passed, "cost_usd": None if any(cost is None for cost in costs) else sum(costs)}
        return check

    return {name: make(steps) for name, steps in checks.items()}
