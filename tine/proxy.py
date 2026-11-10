"""Development OpenAI/Anthropic proxy with tenant auth and a local ledger.

Runtime routing defaults to the frontier model. Offline calibration artifacts are
not loaded implicitly. Optional, explicitly allowlisted client model choices are
useful for development; they are not a learned router or a quality guarantee.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat

from .ledger import MAX_TOKEN_COUNT, Ledger


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


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

    def __repr__(self) -> str:
        return "ProxyConfig(<credentials and configuration redacted>)"

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        """Read configuration, never print or persist credential values."""
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
        )

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


class OutcomeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    success: StrictBool
    cost_usd: StrictFloat | None = Field(default=None, ge=0, allow_inf_nan=False)
    baseline_cost_usd: StrictFloat | None = Field(default=None, ge=0, allow_inf_nan=False)


def _usage(provider: str, payload: dict[str, Any]) -> dict[str, int | None]:
    raw = payload.get("usage") or {}
    if not isinstance(raw, dict):
        raw = {}
    if provider == "openai":
        details = raw.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            details = {}
        values = {
            "input_tokens": raw.get("prompt_tokens"),
            "output_tokens": raw.get("completion_tokens"),
            "cached_tokens": details.get("cached_tokens", 0),
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


def _upstream_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    return base + (path[3:] if base.endswith("/v1") and path.startswith("/v1/") else path)


def create_app(
    config: ProxyConfig | None = None,
    *,
    ledger: Ledger | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    config = config or ProxyConfig.from_env()
    config.validate()
    store = ledger if ledger is not None else Ledger(config.ledger_path)
    upstream = client if client is not None else httpx.AsyncClient(timeout=config.timeout_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            try:
                if client is None:
                    await upstream.aclose()
            finally:
                if ledger is None:
                    store.close()

    app = FastAPI(title="Tine development proxy", lifespan=lifespan)
    app.state.ledger = store

    def record_safely(**values) -> bool:
        # Persistence failure must not discard a provider's answer. The response
        # header makes the missing audit record visible without leaking DB data.
        try:
            store.record(**values)
            return True
        except Exception:
            return False

    async def tenant(request: Request) -> str:
        authorization = request.headers.get("authorization", "")
        token = authorization[7:] if authorization.startswith("Bearer ") else request.headers.get("x-api-key", "")
        for candidate, identity in config.tenant_keys.items():
            if hmac.compare_digest(token.encode("utf-8"), candidate.encode("utf-8")):
                return identity
        raise HTTPException(status_code=401, detail="Invalid tenant credentials")

    @app.get("/health")
    async def health():
        return {"status": "ok", "routing": "frontier-default", "storage": "sqlite-development"}

    @app.patch("/v1/outcomes/{request_id}")
    async def outcome(request_id: str, update: OutcomeUpdate, identity: str = Depends(tenant)):
        fields = update.model_dump(exclude_unset=True)
        success = fields.pop("success")
        if not store.update_outcome(identity, request_id, success, **fields):
            raise HTTPException(status_code=404, detail="Request not found")
        return {"request_id": request_id, "success": success}

    @app.get("/v1/savings")
    async def savings(identity: str = Depends(tenant)):
        return store.report(identity)

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
                    tenant=identity, provider=provider, model=selected,
                    requested_model=requested, request_body=body, cache_context=cache_context,
                    response_body=cached["response_body"], request_id=request_id,
                    upstream_status=cached["upstream_status"], cost_usd=0.0,
                    baseline_cost_usd=cached["baseline_cost_usd"],
                    cached_from=cached["request_id"],
                    latency_ms=(time.monotonic() - started) * 1000,
                )
                return JSONResponse(
                    cached["response_body"], status_code=cached["upstream_status"],
                    headers={"x-tine-request-id": request_id, "x-tine-cache": "hit", "x-tine-ledger": "recorded" if recorded else "unavailable"},
                )

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
            stream=stream, fallback_used=fallback, cache_context=cache_context,
        )
        if response is None:
            recorded = record_safely(**common, latency_ms=(time.monotonic() - started) * 1000)
            return JSONResponse(
                {"error": {"message": "Upstream request failed", "type": "upstream_error"}},
                status_code=502, headers={"x-tine-request-id": request_id, "x-tine-ledger": "recorded" if recorded else "unavailable"},
            )

        result_headers = {"x-tine-request-id": request_id, "x-tine-model": selected}
        for key in ("retry-after", "request-id", "x-request-id"):
            if key in response.headers:
                result_headers[key] = response.headers[key]

        if stream and response.status_code < 400:
            # Streaming never enters the verified answer cache. Costs stay unknown
            # until actual usage/cost is supplied through the outcome endpoint.
            recorded = record_safely(**common, upstream_status=response.status_code)
            result_headers["x-tine-ledger"] = "recorded" if recorded else "unavailable"

            async def chunks():
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                finally:
                    await response.aclose()

            return StreamingResponse(
                chunks(), status_code=response.status_code,
                media_type=response.headers.get("content-type", "text/event-stream"),
                headers=result_headers,
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
            latency_ms=(time.monotonic() - started) * 1000,
        )
        result_headers["x-tine-ledger"] = "recorded" if recorded else "unavailable"
        return Response(
            raw, status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
            headers=result_headers,
        )

    @app.post("/v1/chat/completions")
    async def openai(request: Request, identity: str = Depends(tenant)):
        return await forward(request, "openai", identity)

    @app.post("/v1/messages")
    async def anthropic(request: Request, identity: str = Depends(tenant)):
        return await forward(request, "anthropic", identity)

    return app
