"""Development OpenAI/Anthropic proxy with tenant auth and a local ledger.

Runtime routing defaults to the frontier model. Nothing offline is loaded
implicitly: a frozen policy (tine.policy) runs only when the operator configures it,
either in shadow mode (report only) or as a route policy that serves answers, with
named checks from tine.runtime_checks. Optional, explicitly allowlisted client model
choices are useful for development; they are not a learned router or a quality
guarantee.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat
from starlette.background import BackgroundTask

from .ledger import MAX_TOKEN_COUNT, HoldoutCostError, Ledger, MeasuredBaselineError, MeasuredCostError
from .policy import FREEZE_EDITED, freeze_state, run_stages_async
from .runtime_checks import build_checks, token_variables, validate_checks
from .shadow import (
    TOOLS_SKIP_REASON, NoAnswer, ShadowStore, extract_answer, has_answer, plain_finish, sampled, shadow_entries,
    shadow_stages, skips_for_tools, validate_shadow_policy,
)

log = logging.getLogger("tine.proxy")


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _json_file(variable: str) -> tuple[Any, bytes | None]:
    """The JSON file an environment variable names, and its bytes; ``({}, None)`` when
    the variable is unset. An error names the variable and the path."""
    path = os.environ.get(variable)
    if not path:
        return {}, None
    try:
        raw = Path(path).read_bytes()
        return json.loads(raw.decode("utf-8")), raw
    except (OSError, ValueError, RecursionError) as error:
        raise ValueError(f"{variable}={path} could not be loaded as JSON: {type(error).__name__}: {error}") from None


def _has_stages(policies: dict) -> bool:
    """Whether any tenant's policy has a cheaper stage to run."""
    return any(policy["stages"] for by_provider in policies.values() for policy in by_provider.values())


@dataclass(repr=False)
class ProxyConfig:
    tenant_keys: dict[str, str] = field(default_factory=dict)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openai_frontier_model: str = ""
    anthropic_frontier_model: str = ""
    openai_base_url: str = "https://api.openai.com"
    anthropic_base_url: str = "https://api.anthropic.com"
    ledger_path: str = "artifacts/ledger.sqlite3"
    timeout_seconds: float = 30.0
    allowed_models: dict[str, list[str]] = field(default_factory=dict)
    prices: dict[str, dict[str, float]] = field(default_factory=dict)
    verified_cache: bool = True
    # Frozen policies, tenant -> provider -> policy. A shadow policy only reports
    # (off unless a sample rate is set); a route policy serves answers.
    shadow_policies: dict[str, dict[str, dict]] = field(default_factory=dict)
    shadow_sample_rate: float = 0.0
    shadow_max_cost_usd: float | None = None
    route_policies: dict[str, dict[str, dict]] = field(default_factory=dict)
    route_estimate_baseline: bool = False
    # Named checks the proxy may run (tine.runtime_checks), and how a comparable
    # answer is read from a response for agreement stages. ``checks`` is for every
    # tenant; ``tenant_checks`` (tenant -> name -> steps) belongs to one tenant and
    # wins over a shared check of the same name. A policy may carry its own
    # ``answer_pattern``, which wins over this one: it is part of what was frozen.
    # This one is the fallback for policies without.
    checks: dict[str, list] = field(default_factory=dict)
    tenant_checks: dict[str, dict[str, list]] = field(default_factory=dict)
    answer_pattern: str | None = None
    # Shadow runs in flight at once, for the whole proxy, and optionally for any one
    # tenant (None: no limit of its own), so that one busy tenant cannot take every
    # slot. A sampled request that arrives when either limit is reached is not
    # shadowed; it is counted as skipped in the tenant's shadow report.
    shadow_max_concurrent: int = 8
    shadow_max_concurrent_per_tenant: int | None = None
    # Shadowing stops for a tenant when its shadow calls and checks of unknown cost
    # (the spending cap cannot count them) are more than this number and also more
    # than this share of all its shadow calls and checks.
    shadow_max_unknown_cost_calls: int = 20
    shadow_max_unknown_cost_share: float = 0.2
    # How long a route policy's cheaper stages may take together before the request
    # goes to the frontier. None means timeout_seconds.
    route_deadline_seconds: float | None = None
    # When true, a request whose ledger row cannot be written gets a 503 instead of an
    # answer that would go unrecorded.
    ledger_required: bool = False
    # SHA-256 of the route policies file's bytes, read once at startup and written on
    # every row a route policy ran for. None for a configuration built in code: the
    # proxy then hashes the policies' canonical JSON.
    route_policies_sha256: str | None = None
    # The share of requests a route policy could serve that skip it and go to the
    # frontier instead, chosen from the request id. Their costs against the routed
    # requests' give the saving under live routing (``holdout_estimate`` in the ledger
    # report). 0 turns the holdout off.
    audit_rate: float = 0.0

    def __repr__(self) -> str:
        return "ProxyConfig(<credentials and configuration redacted>)"

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        """Read configuration, never print or persist credential values."""
        shadow_policies, _ = _json_file("TINE_SHADOW_POLICIES")
        # Hashed from the same bytes that are parsed, so the digest is of what runs.
        route_policies, route_bytes = _json_file("TINE_ROUTE_POLICIES")
        return cls(
            tenant_keys=json.loads(os.environ.get("TINE_TENANT_KEYS", "{}")),
            openai_api_key=os.environ.get("TINE_OPENAI_API_KEY", ""),
            anthropic_api_key=os.environ.get("TINE_ANTHROPIC_API_KEY", ""),
            openai_frontier_model=os.environ.get("TINE_OPENAI_FRONTIER_MODEL", ""),
            anthropic_frontier_model=os.environ.get("TINE_ANTHROPIC_FRONTIER_MODEL", ""),
            openai_base_url=os.environ.get("TINE_OPENAI_BASE_URL", "https://api.openai.com"),
            anthropic_base_url=os.environ.get(
                "TINE_ANTHROPIC_BASE_URL", "https://api.anthropic.com"
            ),
            ledger_path=os.environ.get("TINE_LEDGER_PATH", "artifacts/ledger.sqlite3"),
            timeout_seconds=float(os.environ.get("TINE_TIMEOUT_SECONDS", "30")),
            allowed_models=json.loads(os.environ.get("TINE_ALLOWED_MODELS", "{}")),
            prices=json.loads(os.environ.get("TINE_MODEL_PRICES", "{}")),
            verified_cache=os.environ.get("TINE_VERIFIED_CACHE", "true").lower() == "true",
            shadow_policies=shadow_policies,
            shadow_sample_rate=float(os.environ.get("TINE_SHADOW_SAMPLE_RATE", "0")),
            shadow_max_cost_usd=(float(os.environ["TINE_SHADOW_MAX_COST_USD"])
                                 if os.environ.get("TINE_SHADOW_MAX_COST_USD") else None),
            route_policies=route_policies,
            route_policies_sha256=hashlib.sha256(route_bytes).hexdigest() if route_bytes is not None else None,
            route_estimate_baseline=os.environ.get("TINE_ROUTE_ESTIMATE_BASELINE", "false").lower() == "true",
            checks=_json_file("TINE_CHECKS")[0],
            tenant_checks=_json_file("TINE_TENANT_CHECKS")[0],
            answer_pattern=os.environ.get("TINE_ANSWER_PATTERN") or None,
            shadow_max_concurrent=int(os.environ.get("TINE_SHADOW_MAX_CONCURRENT", "8")),
            shadow_max_concurrent_per_tenant=(int(os.environ["TINE_SHADOW_MAX_CONCURRENT_PER_TENANT"])
                                              if os.environ.get("TINE_SHADOW_MAX_CONCURRENT_PER_TENANT") else None),
            shadow_max_unknown_cost_calls=int(os.environ.get("TINE_SHADOW_MAX_UNKNOWN_COST_CALLS", "20")),
            shadow_max_unknown_cost_share=float(os.environ.get("TINE_SHADOW_MAX_UNKNOWN_COST_SHARE", "0.2")),
            route_deadline_seconds=(float(os.environ["TINE_ROUTE_DEADLINE_SECONDS"])
                                    if os.environ.get("TINE_ROUTE_DEADLINE_SECONDS") else None),
            ledger_required=os.environ.get("TINE_LEDGER_REQUIRED", "").lower() in {"1", "true"},
            audit_rate=float(os.environ.get("TINE_AUDIT_RATE", "0")),
        )

    def checks_for(self, tenant: str) -> dict[str, list]:
        """The named checks one tenant's policies may use: the shared ones, and its own."""
        return {**self.checks, **self.tenant_checks.get(tenant, {})}

    def validate(self) -> None:
        if not isinstance(self.tenant_keys, dict) or not self.tenant_keys or any(
            not isinstance(key, str)
            or not key
            or not isinstance(tenant, str)
            or not tenant
            for key, tenant in self.tenant_keys.items()
        ):
            raise ValueError("Configure nonempty TINE_TENANT_KEYS before starting the proxy")
        if not _finite_number(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not isinstance(self.verified_cache, bool):
            raise ValueError("verified_cache must be a boolean")
        if not isinstance(self.allowed_models, dict) or any(
            provider not in {"openai", "anthropic"}
            or not isinstance(models, list)
            or any(not isinstance(model, str) or not model.strip() for model in models)
            for provider, models in self.allowed_models.items()
        ):
            raise ValueError("TINE_ALLOWED_MODELS must map provider names to lists of nonempty model names")
        for base in (self.openai_base_url, self.anthropic_base_url):
            try:
                if not isinstance(base, str):
                    raise ValueError
                parsed = urlsplit(base)
                # Access validates nonnumeric and out-of-range port strings.
                parsed.port
            except ValueError:
                raise ValueError("Provider base URLs must be valid HTTP(S) URLs without credentials") from None
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Provider base URLs must be HTTP(S) URLs without credentials")
        if (not _finite_number(self.shadow_sample_rate) or isinstance(self.shadow_sample_rate, bool)
                or not 0 <= self.shadow_sample_rate <= 1):
            raise ValueError("shadow_sample_rate must be between 0 and 1")
        if not _finite_number(self.audit_rate) or not 0 <= self.audit_rate <= 1:
            raise ValueError("audit_rate must be between 0 and 1")
        if self.shadow_max_cost_usd is not None and (
                not _finite_number(self.shadow_max_cost_usd) or self.shadow_max_cost_usd < 0):
            raise ValueError("shadow_max_cost_usd must be a finite nonnegative amount")
        if self.answer_pattern is not None:
            try:
                re.compile(self.answer_pattern)
            except (re.error, TypeError):
                raise ValueError("answer_pattern must be a valid regular expression") from None
        if not isinstance(self.route_estimate_baseline, bool):
            raise ValueError("route_estimate_baseline must be a boolean")
        if not isinstance(self.ledger_required, bool):
            raise ValueError("ledger_required must be a boolean")
        if self.route_policies_sha256 is not None and (
                not isinstance(self.route_policies_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.route_policies_sha256)):
            raise ValueError("route_policies_sha256 must be a lowercase hexadecimal SHA-256 digest")
        if self.route_deadline_seconds is not None and (
                not _finite_number(self.route_deadline_seconds) or self.route_deadline_seconds <= 0):
            raise ValueError("route_deadline_seconds must be finite and positive")
        for name, least in (("shadow_max_concurrent", 1), ("shadow_max_unknown_cost_calls", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ValueError(f"{name} must be a whole number of at least {least}")
        per_tenant = self.shadow_max_concurrent_per_tenant
        if per_tenant is not None and (isinstance(per_tenant, bool) or not isinstance(per_tenant, int)
                                       or per_tenant < 1):
            raise ValueError("shadow_max_concurrent_per_tenant must be a whole number of at least 1, or unset")
        if (not _finite_number(self.shadow_max_unknown_cost_share)
                or not 0 <= self.shadow_max_unknown_cost_share <= 1):
            raise ValueError("shadow_max_unknown_cost_share must be between 0 and 1")
        if not isinstance(self.tenant_checks, dict) or any(
                name not in self.tenant_keys.values() for name in self.tenant_checks):
            raise ValueError("tenant_checks must map configured tenants to their named checks")
        try:
            for checks in (self.checks, *self.tenant_checks.values()):
                validate_checks(checks)
        except ValueError as error:
            raise ValueError(f"checks: {error}") from None
        every_step = [step for checks in (self.checks, *self.tenant_checks.values())
                      for steps in checks.values() for step in steps]
        if any(step["type"] == "judge" for step in every_step) and not self.anthropic_api_key:
            raise ValueError("checks: a judge step needs the Anthropic provider key")
        sampling = self.shadow_sample_rate > 0 and bool(self.shadow_policies)
        if sampling and self.shadow_max_cost_usd is None:
            raise ValueError("Shadow mode spends money: set shadow_max_cost_usd before sampling")
        # Tenants whose policies name each shared check, to keep one tenant's
        # requests away from a webhook that another tenant's requests also reach.
        shared_users: dict[str, set[str]] = {}
        for label, policies in (("shadow_policies", self.shadow_policies), ("route_policies", self.route_policies)):
            if not isinstance(policies, dict):
                raise ValueError(f"{label} must map tenants to provider policies")
            for name, by_provider in policies.items():
                if name not in self.tenant_keys.values() or not isinstance(by_provider, dict):
                    raise ValueError(f"{label} must map configured tenants to provider policies")
                for provider, policy in by_provider.items():
                    if provider not in {"openai", "anthropic"}:
                        raise ValueError(f"{label} providers must be openai or anthropic")
                    try:
                        validate_shadow_policy(policy, getattr(self, f"{provider}_frontier_model"),
                                               self.checks_for(name))
                    except ValueError as error:
                        raise ValueError(f"{label}: {error}") from None
                    if policy.get("tenant") != name:
                        raise ValueError(f"{label}: a policy must name the tenant it was frozen for, "
                                         "and cannot be used for another tenant")
                    if (label == "route_policies" and any(stage["check"] == "agreement" for stage in policy["stages"])
                            and not (policy.get("answer_pattern") or self.answer_pattern)):
                        # Whole replies of free text almost never agree: every request would
                        # pay for the cheaper models and then the frontier.
                        raise ValueError(
                            f"route_policies: tenant {name!r}'s {provider} policy has agreement stages but no answer "
                            "reader, so whole reply texts would be compared. Freeze it with "
                            "'tine freeze-policy --answer-pattern REGEX', or set TINE_ANSWER_PATTERN")
                    for stage in policy["stages"]:
                        if stage["check"] in self.checks and stage["check"] not in self.tenant_checks.get(name, {}):
                            shared_users.setdefault(stage["check"], set()).add(name)
                    if sampling and label == "shadow_policies":
                        self._require_shadow_prices(name, provider, policy)
        for check, users in shared_users.items():
            if len(users) > 1 and any(step["type"] == "webhook" for step in self.checks[check]):
                raise ValueError(
                    f"checks: {check!r} has a webhook and is named by policies of several tenants "
                    f"({', '.join(sorted(users))}). A webhook receives a tenant's request: give each of them "
                    "its own entry under tenant_checks")
        # A webhook token belongs to one tenant. One variable sent by the checks of two
        # tenants would hand one tenant's CI credential to the other's webhook.
        token_users: dict[str, set[str]] = {}
        for name, own in self.tenant_checks.items():
            for steps in own.values():
                for variable in token_variables(steps):
                    token_users.setdefault(variable, set()).add(name)
        for check, users in shared_users.items():
            for variable in token_variables(self.checks[check]):
                token_users.setdefault(variable, set()).update(users)
        for variable, users in sorted(token_users.items()):
            if len(users) > 1:
                raise ValueError(
                    f"checks: bearer_env {variable!r} is sent by the webhook checks of several tenants "
                    f"({', '.join(sorted(users))}). Give each tenant's webhook a token variable of its own")
        for name, by_provider in self.route_policies.items():
            if set(by_provider) & set(self.shadow_policies.get(name, {})):
                raise ValueError("A tenant's provider can have a shadow policy or a route policy, not both")
        if not isinstance(self.prices, dict):
            raise ValueError("Model prices must map model names to USD rate objects")
        for rates in self.prices.values():
            if not isinstance(rates, dict):
                raise ValueError("Model prices must map model names to USD rate objects")
            for key, value in rates.items():
                if key not in {
                    "input_per_million",
                    "output_per_million",
                    "cached_input_per_million",
                    "cache_write_per_million",
                } or not _finite_number(value) or value < 0:
                    raise ValueError("Model prices must be explicit finite nonnegative USD rates")

    def _require_shadow_prices(self, tenant: str, provider: str, policy: dict) -> None:
        """Refuse to sample a policy whose shadow calls could not be priced. The
        spending cap counts known costs, so an unpriced model would never reach it."""
        needed = {f"{provider}:{model}" for stage in policy["stages"] for model in stage["models"]}
        needed |= {f"anthropic:{step['model']}" for stage in policy["stages"]
                   for step in self.checks_for(tenant).get(stage["check"], []) if step["type"] == "judge"}
        missing = sorted(key for key in needed if not isinstance(self.prices, dict)
                         or not isinstance(self.prices.get(key), dict)
                         or not {"input_per_million", "output_per_million"} <= set(self.prices[key]))
        if missing:
            raise ValueError("Shadow mode's spending cap counts known costs only: set an input and output "
                             f"price for {missing} before sampling")


class OutcomeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    success: StrictBool
    cost_usd: StrictFloat | None = Field(default=None, ge=0, allow_inf_nan=False)
    routing_cost_usd: StrictFloat | None = Field(default=None, ge=0, allow_inf_nan=False)
    baseline_cost_usd: StrictFloat | None = Field(default=None, ge=0, allow_inf_nan=False)


def _usage(provider: str, payload: dict[str, Any]) -> dict[str, int | None]:
    raw = payload.get("usage") or {}
    if not isinstance(raw, dict):
        raw = {}
    if provider == "openai":
        # Omitted optional details retain the no-cache default. An explicitly
        # malformed/null breakdown is unknown, not evidence of zero cache use.
        details = raw.get("prompt_tokens_details", {})
        values = {
            "input_tokens": raw.get("prompt_tokens"),
            "output_tokens": raw.get("completion_tokens"),
            "cached_tokens": details.get("cached_tokens", 0) if isinstance(details, dict) else None,
            "cache_write_tokens": 0,
        }
    else:
        noncached = raw.get("input_tokens")
        cached = raw.get("cache_read_input_tokens", 0)
        written = raw.get("cache_creation_input_tokens", 0)
        values = {
            "input_tokens": (
                noncached + cached + written
                if all(isinstance(x, int) and not isinstance(x, bool) for x in (noncached, cached, written))
                else None
            ),
            "output_tokens": raw.get("output_tokens"),
            "cached_tokens": cached,
            "cache_write_tokens": written,
        }
    return {
        key: value
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_TOKEN_COUNT
        else None
        for key, value in values.items()
    }


def _cost(config: ProxyConfig, provider: str, model: str, tokens: dict) -> float | None:
    rates = config.prices.get(f"{provider}:{model}")
    if not rates or any(value is None for value in tokens.values()):
        return None
    if "input_per_million" not in rates or "output_per_million" not in rates:
        return None
    cached = tokens["cached_tokens"]
    written = tokens["cache_write_tokens"]
    ordinary = tokens["input_tokens"] - cached - written
    if ordinary < 0:
        return None
    if cached and "cached_input_per_million" not in rates:
        return None
    if written and "cache_write_per_million" not in rates:
        return None
    # Scale the bounded token counts first. An unscaled numerator can overflow
    # even when the final dollar amount is representable.
    cost = (
        (ordinary / 1_000_000) * rates["input_per_million"]
        + (tokens["output_tokens"] / 1_000_000) * rates["output_per_million"]
        + (cached / 1_000_000) * rates.get("cached_input_per_million", 0)
        + (written / 1_000_000) * rates.get("cache_write_per_million", 0)
    )
    return cost if math.isfinite(cost) else None


def _answer_reader(policy: dict, config: ProxyConfig) -> dict:
    """How a policy's agreement stages read a comparable answer, and where that came from."""
    if policy.get("answer_pattern"):
        return {"answer_pattern": policy["answer_pattern"], "source": "policy file"}
    if config.answer_pattern:
        return {"answer_pattern": config.answer_pattern, "source": "TINE_ANSWER_PATTERN"}
    return {"answer_pattern": None, "source": "none: whole reply texts are compared"}


def _policy_summary(policy: dict, config: ProxyConfig) -> dict:
    """What an operator needs to know about a loaded policy, for the log and /v1/shadow."""
    reader = _answer_reader(policy, config)
    summary = {"task_type": policy.get("task_type"), "cheaper_stages": len(policy["stages"]),
               "answer_reader": reader, "freeze_digest": freeze_state(policy)}
    if reader["answer_pattern"] is None and any(stage["check"] == "agreement" for stage in policy["stages"]):
        summary["warning"] = ("agreement stages with no answer reader: whole reply texts are compared, so cheaper "
                              "answers are rarely accepted and the projection is low; set the policy's "
                              "answer_pattern (tine freeze-policy --answer-pattern) or TINE_ANSWER_PATTERN")
    return summary


def _upstream_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    return base + (path[3:] if base.endswith("/v1") and path.startswith("/v1/") else path)


def create_app(
    config: ProxyConfig | None = None,
    *,
    ledger: Ledger | None = None,
    client: httpx.AsyncClient | None = None,
    policy_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    config = config or ProxyConfig.from_env()
    config.validate()
    store = ledger if ledger is not None else Ledger(config.ledger_path)
    upstream = client if client is not None else httpx.AsyncClient(timeout=config.timeout_seconds)
    # Everything a policy does (cheaper stages, webhooks, judge calls, in shadow mode
    # and in live routing) uses connections of its own. A slow webhook or cheap model
    # can then use up only these, never the ones that carry frontier requests.
    own_side = policy_client is None and client is None
    side = (policy_client if policy_client is not None
            else client if client is not None else httpx.AsyncClient(timeout=config.timeout_seconds))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            try:
                if client is None:
                    await upstream.aclose()
            finally:
                try:
                    if own_side:
                        await side.aclose()
                finally:
                    if ledger is None:
                        store.close()

    app = FastAPI(title="Tine development proxy", lifespan=lifespan)
    app.state.ledger = store
    shadow_on = bool(config.shadow_policies) and config.shadow_sample_rate > 0
    shadow_store = ShadowStore(store) if shadow_on else None
    shadow_running = 0
    shadow_running_by_tenant: dict[str, int] = {}
    route_deadline = (config.route_deadline_seconds if config.route_deadline_seconds is not None
                      else config.timeout_seconds)
    # One line per loaded policy, so an operator can see which policy runs on which
    # tenant's traffic, of what kind, and how its answers are read.
    for label, policies in (("shadow", config.shadow_policies), ("route", config.route_policies)):
        for name, by_provider in policies.items():
            for provider, policy in by_provider.items():
                summary = _policy_summary(policy, config)
                reader = summary["answer_reader"]
                log.info("%s policy for tenant %r on %s: task_type %r, %d cheaper stage(s), answer reader %s (%s), "
                         "policy file %s", label, name, provider, summary["task_type"], summary["cheaper_stages"],
                         repr(reader["answer_pattern"]) if reader["answer_pattern"] else "none", reader["source"],
                         summary["freeze_digest"])
                if "warning" in summary:
                    log.warning("%s policy for tenant %r on %s: %s", label, name, provider, summary["warning"])
                if summary["freeze_digest"] == FREEZE_EDITED:
                    log.warning("%s policy for tenant %r on %s was edited after freezing: its content does not "
                                "match the digest tine freeze-policy wrote, so its guarantee is only what the file "
                                "says", label, name, provider)
    # Which route policies a routed ledger row was served under. Computed once: the
    # policies are read only at startup.
    policy_sha256 = config.route_policies_sha256 or (hashlib.sha256(json.dumps(
        config.route_policies, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        if config.route_policies else None)
    # What /health reports: a policy file with no stages routes nothing.
    routing_mode = "policy" if _has_stages(config.route_policies) else "frontier-default"
    shadow_mode = "sampling" if shadow_on and _has_stages(config.shadow_policies) else "off"

    def record_safely(**values) -> bool:
        # By default a persistence failure must not discard a provider's answer. The
        # response header makes the missing audit record visible without leaking DB
        # data; with ledger_required the caller gets a 503 instead (``unrecorded``).
        try:
            store.record(**values)
            return True
        except Exception:
            return False

    def unrecorded(request_id: str) -> JSONResponse:
        # With ledger_required, an answer that cannot be recorded is not served.
        return JSONResponse(
            {"error": {"message": "The request could not be recorded in the ledger", "type": "ledger_unavailable"}},
            status_code=503, headers={"x-tine-request-id": request_id, "x-tine-ledger": "unavailable"})

    async def tenant(request: Request) -> str:
        authorization = request.headers.get("authorization", "")
        token = authorization[7:] if authorization.startswith("Bearer ") else request.headers.get("x-api-key", "")
        for candidate, identity in config.tenant_keys.items():
            if hmac.compare_digest(token.encode("utf-8"), candidate.encode("utf-8")):
                return identity
        raise HTTPException(status_code=401, detail="Invalid tenant credentials")

    @app.get("/health")
    async def health():
        return {"status": "ok", "routing": routing_mode, "storage": "sqlite-development", "shadow": shadow_mode}

    @app.patch("/v1/outcomes/{request_id}")
    async def outcome(request_id: str, update: OutcomeUpdate, identity: str = Depends(tenant)):
        fields = update.model_dump(exclude_unset=True)
        success = fields.pop("success")
        try:
            found = store.update_outcome(identity, request_id, success, **fields)
        except MeasuredBaselineError:
            raise HTTPException(status_code=409, detail="The proxy measured this request's baseline cost; "
                                                        "it cannot be replaced") from None
        except MeasuredCostError:
            raise HTTPException(status_code=409, detail="The proxy measured this request's baseline cost and "
                                                        "recorded its costs; a recorded cost cannot be replaced") from None
        except HoldoutCostError:
            raise HTTPException(status_code=409, detail="This request is in the frontier holdout comparison, which "
                                                        "rests on the costs the proxy recorded; its costs cannot be "
                                                        "set") from None
        if not found:
            raise HTTPException(status_code=404, detail="Request not found")
        return {"request_id": request_id, "success": success}

    @app.get("/v1/savings")
    async def savings(identity: str = Depends(tenant)):
        return store.report(identity)

    @app.get("/v1/shadow")
    async def shadow_report(identity: str = Depends(tenant)):
        if shadow_store is None:
            raise HTTPException(status_code=404, detail="Shadow mode is not configured")
        sampling = (shadow_store.state(identity, config.shadow_max_cost_usd, config.shadow_max_unknown_cost_calls,
                                       config.shadow_max_unknown_cost_share)
                    if identity in config.shadow_policies else "no shadow policy for this tenant")
        policies = {provider: _policy_summary(policy, config)
                    for provider, policy in config.shadow_policies.get(identity, {}).items()}
        # Differing answers are checked against the strictest quality limit among the
        # tenant's shadow policies: the report adds all of its shadowed requests together.
        limits = [policy["guarantee"]["epsilon"] for policy in config.shadow_policies.get(identity, {}).values()
                  if policy["stages"] and _finite_number(policy["guarantee"].get("epsilon"))]
        return {**shadow_store.report(identity, epsilon=min(limits, default=None)), "sampling": sampling,
                "policies": policies}

    def price(provider: str, model: str, payload) -> float | None:
        return _cost(config, provider, model, _usage(provider, payload)) if isinstance(payload, dict) else None

    def answer_pattern(policy: dict) -> str | None:
        """How answers are read for this policy: its own pattern, else the proxy's."""
        return policy.get("answer_pattern") or config.answer_pattern

    def stage_tools(identity: str, provider: str, policy: dict, body: dict, url: str, headers: dict):
        """How a policy's cheaper stages call a model and run a named check for one request."""
        async def call(model: str) -> dict:
            response = await side.post(url, json={**body, "model": model}, headers=headers)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("provider reply is not an object")
            # NaN and Infinity are not JSON. Such a reply is neither served nor priced.
            json.dumps(payload, allow_nan=False)
            cost = price(provider, model, payload)
            if not has_answer(provider, payload):
                # Empty, an error body with status 200, filtered or refused: paid for,
                # but not an answer. No stage may serve it, whatever its check.
                raise NoAnswer(cost)
            # A reply with no comparable answer was still paid for: keep its cost and
            # let an agreement stage fail on the missing answer. A reply that calls a
            # tool or was cut off has no final answer to compare.
            answer = extract_answer(provider, payload, answer_pattern(policy)) if plain_finish(provider, payload) else None
            return {"answer": answer, "model": model, "cost_usd": cost, "payload": payload, "raw": response.content}

        named = config.checks_for(identity)
        checks = build_checks(
            named, upstream=side, tenant=identity, provider=provider, body=body,
            anthropic_url=_upstream_url(config.anthropic_base_url, "/v1/messages"),
            anthropic_key=config.anthropic_api_key, cost_of=price) if named else {}
        return call, checks

    async def run_shadow(*, request_id: str, identity: str, provider: str, policy: dict, body: dict,
                         url: str, headers: dict, frontier_payload: dict, frontier_cost: float | None) -> None:
        """Run the policy's cheaper stages for a request that was already answered.
        Nothing here can reach the caller: every failure is contained, the run uses the
        policy path's own connections, and the number of runs in flight is bounded.

        A run starts only if the tenant's cap has room for it (``ShadowStore.admit``),
        and what it spent is counted when it ends even if its row cannot be written.
        A sampled request that is not shadowed is counted with the reason.
        """
        nonlocal shadow_running
        mine = shadow_running_by_tenant.get(identity, 0)
        if shadow_running >= config.shadow_max_concurrent or (
                config.shadow_max_concurrent_per_tenant is not None and mine >= config.shadow_max_concurrent_per_tenant):
            count_safely(identity, "skipped_busy")
            return
        try:
            limits = (config.shadow_max_cost_usd, config.shadow_max_unknown_cost_calls,
                      config.shadow_max_unknown_cost_share)
            if shadow_store.state(identity, *limits) != "on":
                count_safely(identity, "skipped_stopped")
                return
            if not shadow_store.admit(identity, *limits):
                # The cap's room is held by runs in flight, or no run has finished yet
                # to say what one costs.
                count_safely(identity, "skipped_busy")
                return
        except Exception:
            count_safely(identity, "failed")
            return
        shadow_running += 1
        shadow_running_by_tenant[identity] = mine + 1
        entries: list[dict] = []
        try:
            call, checks = stage_tools(identity, provider, policy, body, url, headers)
            outcome = await shadow_stages(policy, call, checks)
            entries = shadow_entries(outcome)
            shadow_store.record(
                request_id=request_id, tenant=identity, provider=provider, policy=policy, outcome=outcome,
                frontier_answer=extract_answer(provider, frontier_payload, answer_pattern(policy)),
                frontier_cost_usd=frontier_cost)
        except Exception:
            count_safely(identity, "failed")
            return
        finally:
            shadow_running -= 1
            shadow_running_by_tenant[identity] -= 1
            shadow_store.settle(identity, entries)

    def count_safely(identity: str, name: str) -> None:
        # A count that cannot be written must not reach the caller or the shadow run.
        try:
            shadow_store.count(identity, name)
        except Exception:
            pass

    async def record_skipped(*, request_id: str, identity: str, provider: str, policy: dict,
                             frontier_cost: float | None, reason: str) -> None:
        """A sampled request the policy would send straight to the frontier. Nothing is
        called; the run is stored so the projection counts it at the frontier's cost.
        Once shadowing has stopped it is counted as skipped, like any other sampled request."""
        try:
            if shadow_store.state(identity, config.shadow_max_cost_usd, config.shadow_max_unknown_cost_calls,
                                  config.shadow_max_unknown_cost_share) != "on":
                count_safely(identity, "skipped_stopped")
                return
            shadow_store.record(
                request_id=request_id, tenant=identity, provider=provider, policy=policy,
                outcome={"accepted": False, "stage": None, "served_by": None, "answer": None, "calls": [],
                         "checks": []},
                frontier_answer=None, frontier_cost_usd=frontier_cost, skip_reason=reason)
        except Exception:
            count_safely(identity, "failed")

    async def forward(request: Request, provider: str, identity: str):
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="Request must be JSON") from None
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            raise HTTPException(status_code=400, detail="A messages array is required")
        try:
            json.dumps(body, allow_nan=False, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            raise HTTPException(status_code=400, detail="JSON strings must be valid Unicode") from None
        except ValueError:
            raise HTTPException(status_code=400, detail="JSON numbers must be finite") from None
        if "stream" in body and not isinstance(body["stream"], bool):
            raise HTTPException(status_code=400, detail="stream must be a boolean")
        requested = body.get("model")
        if requested is not None and (not isinstance(requested, str) or not requested):
            raise HTTPException(status_code=400, detail="model must be a nonempty string")
        frontier = getattr(config, f"{provider}_frontier_model")
        api_key = getattr(config, f"{provider}_api_key")
        if not frontier or not api_key:
            raise HTTPException(status_code=503, detail="Provider credentials and frontier model are not configured")
        selected = requested if requested in config.allowed_models.get(provider, []) else frontier
        body["model"] = selected
        stream = body.get("stream", False)
        request_id = str(uuid.uuid4())
        started = time.monotonic()

        headers = {"content-type": "application/json"}
        if provider == "openai":
            headers["authorization"] = f"Bearer {api_key}"
            path = "/v1/chat/completions"
        else:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = request.headers.get("anthropic-version", "2023-06-01")
            if "anthropic-beta" in request.headers:
                headers["anthropic-beta"] = request.headers["anthropic-beta"]
            try:
                headers["anthropic-version"].encode("ascii")
                headers.get("anthropic-beta", "").encode("ascii")
            except UnicodeEncodeError:
                raise HTTPException(status_code=400, detail="Provider protocol headers must contain ASCII text") from None
            path = "/v1/messages"
        url = _upstream_url(getattr(config, f"{provider}_base_url"), path)
        # Only non-secret settings that affect request semantics enter the key.
        cache_context = {
            "url": url,
            "headers": {key: headers[key] for key in ("anthropic-version", "anthropic-beta") if key in headers},
        }

        if config.verified_cache and not stream:
            try:
                cached = store.cached(identity, provider, body, cache_context=cache_context)
                if cached is not None:
                    # Older ledger entries may contain JSON strings that cannot
                    # be rendered as UTF-8 by the cache response serializer.
                    json.dumps(cached["response_body"], allow_nan=False, ensure_ascii=False).encode("utf-8")
            except Exception:
                cached = None
            if cached is not None:
                recorded = record_safely(
                    # The model whose answer this is: a cheaper one if a policy served the source.
                    tenant=identity, provider=provider, model=cached["model"],
                    requested_model=requested, request_body=body, cache_context=cache_context,
                    response_body=cached["response_body"], request_id=request_id,
                    upstream_status=cached["upstream_status"], cost_usd=0.0,
                    baseline_cost_usd=cached["baseline_cost_usd"],
                    baseline_estimated=bool(cached.get("baseline_estimated")),
                    baseline_source=cached.get("baseline_source"),
                    cached_from=cached["request_id"],
                    latency_ms=(time.monotonic() - started) * 1000,
                )
                if not recorded and config.ledger_required:
                    return unrecorded(request_id)
                return JSONResponse(
                    cached["response_body"], status_code=cached["upstream_status"],
                    headers={"x-tine-request-id": request_id, "x-tine-cache": "hit", "x-tine-ledger": "recorded" if recorded else "unavailable"},
                )

        # Live routing: try the tenant's frozen policy before the frontier. Anything that
        # goes wrong here leaves the request on the ordinary frontier path below.
        routing_spent: float | None = None
        route = config.route_policies.get(identity, {}).get(provider)
        routable = route is not None and bool(route["stages"]) and not stream and selected == frontier
        # The frontier holdout: with an audit rate, every request the policy could serve
        # is in one of two groups. A held-out one skips the policy and goes to the
        # frontier with nothing spent on routing. The salt keeps this choice apart from
        # the shadow sample, which reads the same request id.
        holdout = (sampled(f"holdout:{request_id}", config.audit_rate)
                   if routable and config.audit_rate > 0 else None)
        # A request with tool definitions goes straight to the frontier when the policy
        # has an agreement stage: cheaper models cannot agree on tool calls. Nothing is
        # spent on the way, and the ledger row says why.
        route_skip = TOOLS_SKIP_REASON if routable and not holdout and skips_for_tools(route, body) else None
        if route_skip is not None or holdout:
            routing_spent = 0.0
        # Only a policy whose stages were tried traces its row to the policy file.
        routing_ran = routable and route_skip is None and not holdout
        if routing_ran:
            try:
                call, named_checks = stage_tools(identity, provider, route, body, url, headers)
                routed = await run_stages_async(route, call, checks=named_checks, deadline_seconds=route_deadline)
                spent = [entry["cost_usd"] for entry in (*routed["calls"], *routed["checks"])]
                total = None if any(value is None for value in spent) else math.fsum(spent)
            except Exception:
                routed = None
            if routed is not None:
                if routed["accepted"]:
                    served, result = routed["served_by"], routed["result"]
                    answer_cost = result["cost_usd"]
                    tokens = _usage(provider, result["payload"])
                    # What the frontier would have cost is not observed. With the
                    # explicit setting it is estimated from this answer's token counts
                    # at the frontier's prices, and stored and labelled as an estimate;
                    # otherwise it stays unknown.
                    estimate = _cost(config, provider, frontier, tokens) if config.route_estimate_baseline else None
                    recorded = record_safely(
                        # Recorded under the request as it arrived, so that a verified
                        # answer is found by the cache lookup above; the model column
                        # says which model answered.
                        tenant=identity, provider=provider, model=served, requested_model=requested,
                        request_body=body, request_id=request_id, cache_context=cache_context,
                        response_body=result["payload"], upstream_status=200,
                        input_tokens=tokens["input_tokens"], output_tokens=tokens["output_tokens"],
                        cached_tokens=tokens["cached_tokens"], cost_usd=answer_cost,
                        routing_cost_usd=(total - answer_cost if total is not None and answer_cost is not None else None),
                        baseline_cost_usd=estimate, baseline_estimated=estimate is not None,
                        policy_sha256=policy_sha256, route_stage=routed["stage"], holdout=holdout,
                        latency_ms=(time.monotonic() - started) * 1000)
                    if not recorded and config.ledger_required:
                        return unrecorded(request_id)
                    return Response(
                        result["raw"], status_code=200, media_type="application/json",
                        headers={"x-tine-request-id": request_id, "x-tine-model": served,
                                 "x-tine-route": f"stage-{routed['stage']}",
                                 **({"x-tine-baseline": "estimated"} if estimate is not None else {}),
                                 "x-tine-ledger": "recorded" if recorded else "unavailable"})
                routing_spent = total if routed["calls"] else 0.0

        fallback = False
        response = None
        for attempt in range(2):
            try:
                outgoing = upstream.build_request("POST", url, json=body, headers=headers)
                response = await upstream.send(outgoing, stream=stream)
                if response.status_code < 400 or selected == frontier or attempt == 1:
                    # Error bodies are buffered even for streaming requests.
                    # Keep their transport failures inside the upstream handler.
                    if stream and response.status_code >= 400:
                        await response.aread()
                    break
                await response.aclose()
            except httpx.HTTPError:
                if response is not None:
                    await response.aclose()
                response = None
                if selected == frontier or attempt == 1:
                    break
            selected = frontier
            body["model"] = frontier
            fallback = True

        common = dict(
            tenant=identity, provider=provider, model=selected,
            requested_model=requested, request_body=body, request_id=request_id,
            stream=stream, fallback_used=fallback, cache_context=cache_context, route_skip_reason=route_skip,
            holdout=holdout,
            # A route policy ran and sent the request on to the frontier.
            **({"policy_sha256": policy_sha256, "route_stage": "frontier"} if routing_ran else {}),
            # A held-out request names the policy it was held out from; no stage ran.
            **({"policy_sha256": policy_sha256} if holdout else {}),
        )
        if response is None:
            # What a route policy spent before the frontier failed is still known. No
            # answer is served here, so ledger_required does not change the reply.
            recorded = record_safely(**common, routing_cost_usd=routing_spent,
                                     latency_ms=(time.monotonic() - started) * 1000)
            return JSONResponse(
                {"error": {"message": "Upstream request failed", "type": "upstream_error"}},
                status_code=502, headers={"x-tine-request-id": request_id, "x-tine-ledger": "recorded" if recorded else "unavailable"},
            )

        result_headers = {"x-tine-request-id": request_id, "x-tine-model": selected}
        for key in ("retry-after", "request-id", "x-request-id"):
            if key in response.headers:
                result_headers[key] = response.headers[key]

        shadow_policy = config.shadow_policies.get(identity, {}).get(provider) if shadow_on else None
        # Whether a shadow run would have been started for this request, had it not streamed.
        shadow_due = (shadow_policy is not None and shadow_policy["stages"] and selected == frontier
                      and not fallback and response.status_code < 400 and sampled(request_id, config.shadow_sample_rate))
        if stream and response.status_code < 400:
            # Streaming never enters the verified answer cache. Costs stay unknown
            # until actual usage/cost is supplied through the outcome endpoint.
            recorded = record_safely(**common, upstream_status=response.status_code)
            if not recorded and config.ledger_required:
                await response.aclose()
                return unrecorded(request_id)
            result_headers["x-tine-ledger"] = "recorded" if recorded else "unavailable"
            # Shadow mode does not run streamed requests. A sampled one is counted, so
            # that the shadow report says how much of the sample it could not see.
            streamed = BackgroundTask(count_safely, identity, "streamed") if shadow_due else None

            async def chunks():
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                finally:
                    await response.aclose()

            return StreamingResponse(
                chunks(), status_code=response.status_code,
                media_type=response.headers.get("content-type", "text/event-stream"),
                headers=result_headers, background=streamed,
            )

        raw = await response.aread()
        await response.aclose()
        try:
            payload = json.loads(raw)
            json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
        except (ValueError, UnicodeError):
            # Preserve upstream bytes and the request record, but never cache
            # malformed JSON values or derive accounting from them.
            payload = None
        tokens = _usage(provider, payload) if isinstance(payload, dict) else {
            "input_tokens": None, "output_tokens": None, "cached_tokens": None,
            "cache_write_tokens": None,
        }
        cost = _cost(config, provider, selected, tokens) if response.status_code < 400 and not fallback else None
        recorded = record_safely(
            **common, response_body=payload if isinstance(payload, dict) else None,
            upstream_status=response.status_code,
            input_tokens=tokens["input_tokens"], output_tokens=tokens["output_tokens"],
            cached_tokens=tokens["cached_tokens"], cost_usd=cost,
            baseline_cost_usd=cost if selected == frontier else None,
            # Set only when a route policy ran and escalated: what its cheaper
            # attempts and checks cost. Otherwise unknown, as before.
            routing_cost_usd=routing_spent,
            latency_ms=(time.monotonic() - started) * 1000,
        )
        if not recorded and config.ledger_required:
            return unrecorded(request_id)
        if route is not None and route["stages"] and not stream:
            result_headers["x-tine-route"] = "holdout" if holdout else "frontier"
        if route_skip is not None:
            result_headers["x-tine-route-skipped"] = "tools"
        result_headers["x-tine-ledger"] = "recorded" if recorded else "unavailable"
        background = None
        if shadow_due and isinstance(payload, dict):
            # After the response has been sent: the caller never waits for shadow calls.
            if skips_for_tools(shadow_policy, body):
                background = BackgroundTask(
                    record_skipped, request_id=request_id, identity=identity, provider=provider,
                    policy=shadow_policy, frontier_cost=cost, reason=TOOLS_SKIP_REASON)
            else:
                background = BackgroundTask(
                    run_shadow, request_id=request_id, identity=identity, provider=provider, policy=shadow_policy,
                    body=dict(body), url=url, headers=dict(headers), frontier_payload=payload, frontier_cost=cost)
        return Response(
            raw, status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
            headers=result_headers, background=background,
        )

    @app.post("/v1/chat/completions")
    async def openai(request: Request, identity: str = Depends(tenant)):
        return await forward(request, "openai", identity)

    @app.post("/v1/messages")
    async def anthropic(request: Request, identity: str = Depends(tenant)):
        return await forward(request, "anthropic", identity)

    return app
