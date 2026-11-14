"""The patch judge: its instructions, how a reply is read, and one call to it.

A judge reads a GitHub issue and a patch that already passed the free checks and
says whether the patch resolves the issue. The same instructions are used for every
judge that is measured (``scripts/judge_patches.py``) and for the judge a deployed
policy runs (``patch_judge_check`` with ``tine.policy.run_policy``), so a measured
catch rate describes the judge that is deployed. Nothing here executes a patch.

Calling a Claude model costs money and needs the task's authorization (AGENTS.md).
The API key is read from the environment by the SDK client the caller supplies.
"""

from __future__ import annotations

import json
import re

PROMPT_VERSION = "2026-11-11"
INSTRUCTIONS = """You are reviewing a proposed code change before it is accepted.

Below are a GitHub issue from the {repository} repository and a patch that claims to resolve it. The patch already applies cleanly and does not break the project's existing tests. Decide whether it correctly and completely resolves the issue, so that tests written specifically for this issue would pass. You cannot run code; reason from the issue and the patch.

Reply with one JSON object and nothing else:
{{"verdict": "accept" or "reject", "probability": a number from 0 to 1 that the patch correctly resolves the issue, "reason": "at most 20 words"}}"""
# A second style for smaller models, which tend to accept everything when asked for a
# bare verdict: make the reviewer work through the change before deciding.
THINK_INSTRUCTIONS = """You are a strict code reviewer deciding whether a proposed change may be merged.

Below are a GitHub issue from the {repository} repository and a patch that claims to resolve it. The patch applies cleanly and does not break the project's existing tests, but about half of patches like this do NOT actually fix the issue. You cannot run code.

Work through these steps in a few sentences each:
1. What exactly must change for the issue to be resolved?
2. What does the patch change, and where?
3. Name the most likely way this patch fails: a case from the issue it does not handle, the wrong location, or behaviour that differs from what the issue asks for.
4. Decide. Accept only if you found no real problem in step 3.

End your reply with one JSON object on its own line:
{{"verdict": "accept" or "reject", "probability": a number from 0 to 1 that the patch correctly resolves the issue, "reason": "at most 20 words"}}"""
STYLES = {"verdict": None, "think": THINK_INSTRUCTIONS}
CASE_TEMPLATE = "<issue>\n{issue}\n</issue>\n\n<patch>\n{patch}\n</patch>"
MAX_ISSUE_CHARS = 12000
MAX_PATCH_CHARS = 20000
# USD per million tokens (input, output), read from
# https://platform.claude.com/docs/en/about-claude/pricing on 2026-11-14. Output includes the
# model's thinking tokens. Used to stop a run at its spending cap; check before relying on it.
ANTHROPIC_PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0),
                    "claude-opus-5-5": (4.0, 20.0)}


def case_text(case: dict) -> str:
    return CASE_TEMPLATE.format(issue=case["issue"][:MAX_ISSUE_CHARS], patch=case["patch"][:MAX_PATCH_CHARS])


def build_messages(case: dict, style: str = "verdict") -> list[dict]:
    if style not in STYLES:
        raise ValueError(f"style must be one of {sorted(STYLES)}")
    instructions = STYLES[style] or INSTRUCTIONS
    return [{"role": "system", "content": instructions.format(repository=case["repository"])},
            {"role": "user", "content": case_text(case)}]


def parse_verdict(text: str) -> dict:
    """The judge's JSON object from its reply. Raises ValueError if there is none."""
    for candidate in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        verdict = str(value.get("verdict", "")).strip().lower()
        if verdict not in ("accept", "reject"):
            continue
        try:
            probability = float(value.get("probability"))
        except (TypeError, ValueError):
            continue
        if not 0 <= probability <= 1:
            continue
        return {"verdict": verdict, "probability": probability,
                "reason": str(value.get("reason", ""))[:200]}
    raise ValueError("No verdict object with verdict and probability in the reply")


def judge_once(client, case: dict, *, model: str, effort: str = "medium", max_tokens: int = 8000,
               style: str = "verdict", prices: tuple[float, float] | None = None) -> dict:
    """One judge call on a Claude model. ``case`` needs ``repository``, ``issue`` and
    ``patch``. Returns the verdict (``unparsed`` for a refusal or a reply with no
    usable verdict), the tokens the API reported and what the call cost."""
    price_in, price_out = prices or ANTHROPIC_PRICES[model]
    messages = build_messages(case, style)
    response = client.messages.create(
        model=model, max_tokens=max_tokens, system=messages[0]["content"],
        messages=[{"role": "user", "content": messages[1]["content"]}],
        output_config={"effort": effort})
    result = {"prompt_tokens": response.usage.input_tokens, "completion_tokens": response.usage.output_tokens,
              "cost_usd": (response.usage.input_tokens * price_in + response.usage.output_tokens * price_out) / 1e6}
    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        if response.stop_reason == "refusal":
            raise ValueError("refusal")
        result.update(parse_verdict(text))
    except ValueError:
        result.update({"verdict": "unparsed", "probability": None, "reason": str(response.stop_reason)[:200]})
    return result


def worst_case_cost(case: dict, *, model: str, max_tokens: int = 8000, style: str = "verdict",
                    prices: tuple[float, float] | None = None) -> float:
    """An upper estimate of one call's cost, for stopping before a spending cap."""
    price_in, price_out = prices or ANTHROPIC_PRICES[model]
    messages = build_messages(case, style)
    return ((len(messages[0]["content"]) + len(messages[1]["content"])) / 3 * price_in
            + max_tokens * price_out) / 1e6


def patch_judge_check(client, *, model: str, effort: str = "medium", threshold: float | None = None,
                      max_tokens: int = 8000):
    """A check for ``tine.policy.run_policy``: accept a cheaper model's patch when the
    judge does. The result it is given must carry ``repository``, ``issue`` and
    ``patch``. ``threshold`` accepts on probability >= it instead of the verdict. A
    judge call that fails counts as a rejection, so the request escalates."""
    def check(result: dict) -> dict:
        try:
            verdict = judge_once(client, result, model=model, effort=effort, max_tokens=max_tokens)
        except Exception as error:
            return {"passed": False, "cost_usd": None, "detail": type(error).__name__}
        if threshold is None:
            passed = verdict["verdict"] == "accept"
        else:
            passed = verdict["probability"] is not None and verdict["probability"] >= threshold
        return {"passed": passed, "cost_usd": verdict["cost_usd"], "detail": verdict["verdict"]}
    return check
