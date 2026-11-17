# Local proxy and ledger

The development proxy exposes native OpenAI `POST /v1/chat/completions` and Anthropic `POST /v1/messages` routes, including streaming passthrough. It uses an authenticated tenant mapping and sends every request to that provider's configured frontier model unless a frozen policy is configured. With one, it can run the policy in shadow mode (report only) or serve answers from it; see Shadow mode, Routing through a policy and Named checks below, which also list `TINE_SHADOW_POLICIES`, `TINE_SHADOW_SAMPLE_RATE`, `TINE_SHADOW_MAX_COST_USD`, `TINE_SHADOW_MAX_CONCURRENT`, `TINE_SHADOW_MAX_CONCURRENT_PER_TENANT`, `TINE_SHADOW_MAX_UNKNOWN_COST_CALLS`, `TINE_SHADOW_MAX_UNKNOWN_COST_SHARE`, `TINE_SHADOW_MAX_CALLS_PER_MINUTE`, `TINE_ROUTE_POLICIES`, `TINE_ROUTE_ESTIMATE_BASELINE`, `TINE_ROUTE_DEADLINE_SECONDS`, `TINE_AUDIT_RATE`, `TINE_CHECKS`, `TINE_TENANT_CHECKS`, `TINE_ANSWER_PATTERN` and `TINE_ALLOWED_MODELS`. Nothing is enabled by an offline experiment alone.

To run the proxy in a container, see [DEPLOY.md](DEPLOY.md). The image has not been built or run yet.

Use the activated Python environment from the README. Configure these variables through your shell's secure configuration or environment settings:

| Variable | Meaning |
| --- | --- |
| `TINE_TENANT_KEYS` | JSON object mapping incoming proxy tokens to tenant IDs |
| `TINE_OPENAI_API_KEY` / `TINE_ANTHROPIC_API_KEY` | Provider credentials; configure only the providers you use |
| `TINE_OPENAI_FRONTIER_MODEL` / `TINE_ANTHROPIC_FRONTIER_MODEL` | Exact frontier model name for each provider |
| `TINE_LEDGER_PATH` | SQLite file, default `artifacts/ledger.sqlite3`. To copy it while the proxy runs, use `python -m tine backup-ledger --output PATH` ([REPORT.md](REPORT.md)): the main file copied alone can lack the newest rows |
| `TINE_LEDGER_REQUIRED` | `1` or `true` (any case) returns 503 instead of an answer the ledger could not record. Default off |
| `TINE_OPENAI_BASE_URL` / `TINE_ANTHROPIC_BASE_URL` | Optional API-compatible upstreams; defaults are official HTTPS endpoints. Every model a policy names for that provider is called here (see One base URL per provider) |
| `TINE_TIMEOUT_SECONDS` | How long a provider may take over one reply, in seconds. Default 600. A reply that takes longer is a 502 with no retry. It was 30 until 2026-11-17: in the live pilots 12 of about 3,600 frontier replies through the proxy took longer than that (up to 53 seconds). Connecting to the provider is given 10 seconds, or this value if it is lower |
| `TINE_MODEL_PRICES` | Explicit USD-per-million token rates keyed by `provider:model`. Optional, except that shadow sampling needs a price for every stage model and judge model of a shadow policy |
| `TINE_VERIFIED_CACHE` | `true` by default; cache requires an explicit successful outcome |
| `TINE_MAX_BODY_BYTES` | Largest request body on the provider routes, in bytes. Default 10485760 (10 MB). Larger gets 413 (see Request limits) |
| `TINE_DENIED_FIELDS` | Optional JSON object of provider names and arrays of top-level request fields refused with 400. Unset: none |
| `TINE_ANTHROPIC_BETA_ALLOWLIST` | Optional comma-separated `anthropic-beta` values that may be forwarded; others get 400. Unset or empty: all forwarded |

Never commit credentials or paste provider keys into notebook cells. The app refuses startup without configured tenant keys. Clients use their **proxy tenant token**, not the provider key, in `Authorization: Bearer ...`; Anthropic clients can use `x-api-key` for that proxy token. Outbound requests use the server's provider credential. A caller-supplied tenant header cannot select a tenant.

Start locally:

```bash
python -m tine serve --host 127.0.0.1 --port 8000
```

`GET /health` reports the routing mode, whether shadow mode is sampling, and SQLite development storage. `routing` is `policy` only when at least one loaded route policy has a nonempty `stages` list, and `frontier-default` otherwise, including when `TINE_ROUTE_POLICIES` holds only policies with `stages: []`. `shadow` is `sampling` only when the sample rate is above 0, shadow policies are loaded and at least one of them has stages; otherwise it is `off`. Provider routes return a Tine request ID in `x-tine-request-id`. After an objective check completes, authenticate as the same tenant and update it with `PATCH /v1/outcomes/{request_id}` and JSON `{"success": true}` or `{"success": false}`. Optional `cost_usd`, `routing_cost_usd`, and `baseline_cost_usd` must come from actual accounting; they are not guessed. A caller may set a cost the proxy did not record, never replace one it measured (see Where a baseline comes from). `GET /v1/savings` returns that tenant's ledger summary.

Record the returned answer's generation cost in `cost_usd` and the additional routing cost in `routing_cost_usd`. Routing cost includes earlier attempts that escalated, judges, tests, embeddings, and allocated customer-hosted compute. Include each expense once. If an older caller included these expenses in `cost_usd`, split that amount before supplying the routing component. `baseline_cost_usd` is the comparable always-frontier cost.

For example, a verified answer costing $0.25 plus $0.90 in routing against a $1.00 baseline has **-$0.15 net savings**. For a request whose costs and baseline the proxy did not record, send `{"success": true, "cost_usd": 0.25, "routing_cost_usd": 0.90, "baseline_cost_usd": 1.00}`. The baseline is then the caller's, so this saving is reported as `caller_baseline_savings_usd`, not in `verified_savings_usd`. Omitting a cost on later updates preserves it; explicit `null` marks it unknown again. Zero must be an explicit accounting result.

Without a route policy, automatic proxy records leave routing cost unknown, including cache hits: no provider call does not imply free checking or infrastructure. With a route policy the proxy records routing cost itself for requests the policy handled (see Routing through a policy); cache hits still leave it unknown. Cache hits do not copy the source request's routing cost. Older SQLite ledgers gain a nullable routing-cost column; existing rows stay unknown until costs are supplied. Restart all writers together when upgrading. Cache eligibility still depends on the verified outcome, independently of whether costs have arrived.

Net savings require known answer, routing, and baseline costs for a verified successful request. `verified_cost_usd` now includes both cost components, exposed separately as `verified_answer_cost_usd` and `verified_routing_cost_usd`. `verified_savings_usd` and `savings_fraction` are null when no comparable request exists. `unknown_cost_requests` now counts requests missing either component; separate answer/routing counts identify the gaps. Known cost totals sum measured components across successes, failures, and pending requests, so they are incomplete while any component is unknown. Cost per verified success is null until every request has both costs. Unrepresentable numeric aggregates are also null.

Unknown outcomes and costs remain unknown. Cached responses are scoped by tenant, provider, upstream endpoint, model, exact request contents, and forwarded Anthropic version/beta headers. With a route policy the cached answer may be a cheaper model's: a routed answer is stored under the request as it arrived (see Routing through a policy). They become eligible after success is explicitly confirmed. After upgrading, responses recorded with the older cache keys require a fresh upstream request and successful outcome before cache reuse, because those keys lack upstream context. Streaming responses are not cached. The ledger stores prompts/responses; use only data you intend to retain in the configured local database.

Streamed replies are relayed byte for byte. The proxy writes the ledger row before the first byte, then
reads token usage from the events as they pass, without changing them. When the stream ends it fills in
the row's token counts and answer cost (`cost_usd`) from that usage and `TINE_MODEL_PRICES`:

- Anthropic: input tokens (with cache reads and writes) from `message_start`, output tokens from
  `message_delta`.
- OpenAI: the chunk that carries `usage`. OpenAI sends it only when the caller asked for it with
  `"stream_options": {"include_usage": true}`. The proxy does not add that option to a request.
- No usage, or a stream that broke off before the final count: the cost stays unknown, as before. What
  arrived is kept (an Anthropic stream cut off mid-reply keeps its input tokens).
- A stream served after a fallback gets token counts and no cost, as a buffered reply does.
- Only empty fields are filled. A cost the caller sent through the outcome endpoint first is kept.

A streamed row has no baseline: `baseline_cost_usd` and routing cost stay unknown, so streams never
count in `verified_savings_usd`. The caller may still supply both, as for any request the proxy did not
measure. Streams are still never cached, shadowed or routed; `streamed_not_shadowed` counts them as
before. Built and tested with stand-in streams for both providers; never used by a real customer or run
against a real provider's stream.

For OpenAI-compatible usage, omitted optional cache details retain the zero-cache default. Explicitly null or malformed `prompt_tokens_details` instead leave cached tokens, estimated cost, and the automatic frontier baseline unknown, even if input/output totals are valid. The provider response still passes through unchanged; a later measured cost can be supplied through the outcome endpoint.

Late outcome reversals or newer resolved failures invalidate prior cached successes. Resolved outcomes use a transactionally assigned database sequence, so equal or backward-moving wall-clock timestamps cannot revive stale success evidence. Existing ledgers are upgraded automatically; legacy entries retain their prior timestamp order until new sequenced evidence arrives. Restart all proxy writers together when upgrading so every writer uses this ordering. If local ledger recording fails, a successful provider response is still returned with `x-tine-ledger: unavailable`, unless `TINE_LEDGER_REQUIRED` is on (see Audit trail). Net savings reports cover comparable, verified successful requests only; they are not billing amounts or proof of end-to-end savings. Failed and pending requests with known costs contribute to total spend, including routing costs, and to the cost-per-verified-success metric.

`TINE_ALLOWED_MODELS` can explicitly allow client-requested development alternatives, with a single frontier retry on an upstream failure before a response is emitted. Supply a JSON object of provider names and model-name arrays, for example `{"openai": ["EXACT_MODEL_NAME"]}`. Names match exactly. A string in place of an array, an unknown provider key, or an empty model name causes startup to fail before a ledger is opened. This switch is not a calibrated router. Failure after streaming starts cannot safely be retried without duplicating output. Keep it unset for frontier-only operation.

This is a development implementation: SQLite is not the planned production Postgres backend. Tests use mock upstreams, not paid API calls. Do not deploy it as a production guarantee or host it on Colab. Container packaging and its limits are in [DEPLOY.md](DEPLOY.md). No live provider credentials were requested or validated during this work.

### Where a baseline comes from

Each row records where its `baseline_cost_usd` came from, in the `baseline_source` column:

- `proxy_measured`: the proxy called the frontier for this request and priced the reply. A cache hit
  copies the source of the row it was served from.
- `estimated`: written by the proxy with `TINE_ROUTE_ESTIMATE_BASELINE=true` (see Routing through a
  policy).
- `caller`: supplied through `PATCH /v1/outcomes/{id}`.

`verified_savings_usd`, `savings_fraction`, `comparable_verified_requests` and the other `verified_*`
figures use only `proxy_measured` baselines. A baseline the caller sent is reported apart, as
`caller_baseline_requests` and `caller_baseline_savings_usd` (net of routing cost), with
`caller_baseline_scope` saying what the figure rests on. It never counts toward verified savings, in
the same way as the `estimated_baseline_*` figures. `scope` and `savings_scope` say so in words.

A PATCH whose body contains `baseline_cost_usd`, `null` included, for a request whose baseline is
`proxy_measured` returns 409 ("The proxy measured this request's baseline cost; it cannot be
replaced"). Nothing in the request is changed then, not even `success`. A PATCH may still set or
correct a baseline that is missing, estimated or caller-supplied. A `null` baseline clears both the
baseline and its source.

The verified figures also rest on the answer and routing costs of those requests, so the same holds
for them. On a request whose baseline is `proxy_measured`, a PATCH may set `cost_usd` or
`routing_cost_usd` when the proxy left it unknown (routing cost without a route policy, for example),
and may correct a value an earlier PATCH supplied. A PATCH that would replace a cost the proxy
recorded, with a number or with `null`, returns 409 ("The proxy measured this request's baseline cost
and recorded its costs; a recorded cost cannot be replaced") and changes nothing. A cache hit copies
the baseline source of its source row, so its recorded answer cost is protected in the same way. On
other requests the caller may replace either cost: nothing verified rests on them. Whether a cost was
supplied by the caller is read from the outcome history, so a cost set before that history existed
counts as the proxy's.

### Measuring the saving under live routing

Under live routing the proxy calls the frontier only for requests that escalated to it, and those
requests also paid for the cheaper attempts. So the measured saving (`verified_savings_usd`) covers
only escalated requests and cannot come out positive for the policy as a whole. The estimated baseline
is the answer's token counts at the frontier's prices; in the simulated pilots it came to about 0.45
times the frontier's recorded cost, because cheap answers were shorter. Neither figure is the policy's
saving. Measuring that needs a random share of requests always sent to the frontier (a holdout), with a
confidence interval. `TINE_AUDIT_RATE` sets that share. It is off by default (0), and without it the
saving of live routing is a projection from shadow mode or an estimate, never a measurement.

**The frontier holdout.** With `TINE_AUDIT_RATE` between 0 and 1, every request a route policy could
serve (the tenant and provider have a policy with stages, the request does not stream and asks for the
frontier model) falls in one of two groups:

- **Held out**, a share equal to the rate, chosen from the request id as shadow sampling is
  (`sampled("holdout:" + request_id, rate)`; the prefix keeps it apart from the shadow sample). The
  policy is skipped and the frontier answers, tools or not. The response carries `x-tine-route:
  holdout`. The ledger row has `holdout` 1, `routing_cost_usd` 0.0, the frontier's cost as both
  `cost_usd` and a `proxy_measured` baseline, and `policy_sha256` naming the policy it was held out
  from, with `route_stage` NULL.
- **Routed**, the rest: handled exactly as without a holdout (a stage, the frontier, or skipped for
  tools). The row has `holdout` 0.

Rows outside the comparison have `holdout` NULL: no policy, a streamed request, a request for an
allowlisted model, and cache hits (the cache is looked up before the choice is made). Older ledgers
gain the column, NULL on every existing row. Nothing else changes for a held-out request. A tenant's
provider cannot have a shadow policy as well as a route policy, so a held-out request is never shadowed.

`GET /v1/savings` then carries `holdout_estimate` (null while no row is in the comparison):

- `routed_requests` and `holdout_requests`: requests in each group whose answer and routing costs are
  both known.
- `routed_unknown_cost_requests`: routed requests whose answer or routing cost is not known. Routing
  cost is unknown when a cheaper call failed (an HTTP error or a 503), a check raised or had no usage,
  a webhook failed closed, or `TINE_ROUTE_DEADLINE_SECONDS` ran out. Those requests then usually
  escalated and paid the frontier as well, so they are the costliest routed requests. Leaving them out
  would overstate the saving, so each is counted at the part of its cost that is known (unknown parts
  as 0). While any are, `saving_is_upper_limit` is true: the routed mean is a lower limit, and the
  saving and both ends of its interval are upper limits.
- `holdout_unknown_cost_requests`: held-out requests whose cost is not known (a 502, or a frontier model
  with no price). They are left out and counted.
- `routed_mean_cost_usd` (answer plus routing cost, escalations included) and `holdout_mean_cost_usd`.
- `saving_fraction` = 1 - mean(routed) / mean(holdout), with `saving_fraction_low` and
  `saving_fraction_high` at `confidence` 0.95. The interval is a normal approximation on the ratio
  (delta method, the groups independent): the standard error is
  sqrt(var(routed)/n_routed + R^2 var(holdout)/n_holdout) / mean(holdout), with R the ratio, and the
  interval is the saving plus or minus 1.96 of them. `method` says so in words.
- No mean, saving or interval is given until each group has `min_requests_per_group` (30) requests with
  known costs, or while the held-out mean is 0.
- `scope` says what the figure is: an estimate from a random sample, not a billing amount. It assumes
  the two groups are alike, and the random choice is what makes them so. It does not measure quality.

On a request in the comparison (`holdout` 0 or 1), a PATCH that sets `cost_usd` or `routing_cost_usd`
returns 409 ("This request is in the frontier holdout comparison, which rests on the costs the proxy
recorded; its costs cannot be set") and changes nothing. The outcome and a baseline the proxy did not
measure can still be set.

Limits:

- The estimate pools every row in the ledger with `holdout` set, whatever its policy digest or the rate
  at the time. Changing the policy or the rate mixes periods in different proportions in the two
  groups. Start a new ledger, or filter by `policy_sha256` in SQL, after a change.
- At a rate of 5% and 30 held-out requests needed, the first figure needs about 600 eligible requests.
  The interval is wide until both groups are much larger, and it is only approximate for small groups
  or very skewed costs.
- Held-out requests cost the frontier's price. That is part of the price of measuring, and it lowers
  the saving the tenant gets by about the rate times the saving.
- It says nothing about quality. A routed answer that was wrong costs the same here as a right one.
- A routed request whose cost is not fully known turns the figure into an upper limit, and its costs
  cannot be filled in later (the 409 above). Find why routing cost went unknown (a model that fails,
  a webhook, the deadline) and fix it to get a point figure again. Held-out requests whose cost is not
  known are still left out, which can move the figure either way.
- Built and tested with stand-in providers, and run with real current models on public benchmark questions in
  `results/2026-11-17-live-pilot-more` (measured saving MedQA 56%, ARC 97%, FinQA 18%). Never run on a
  customer's traffic.

### Audit trail

- Ledger rows carry `policy_sha256` and `route_stage`.
  - `policy_sha256` is the SHA-256 of the `TINE_ROUTE_POLICIES` file's bytes, read once at startup, so
    `sha256sum route-policies.json` on the deployed file must match it. For a `ProxyConfig` built in
    code it is the SHA-256 of the policies' canonical JSON (`json.dumps` with `sort_keys=True`,
    `separators=(",", ":")`, `ensure_ascii=False`, UTF-8).
  - `route_stage` is text: the stage index (`"0"`, `"1"`, ...) when a cheaper stage answered, matching
    the `x-tine-route: stage-N` header, or `"frontier"` when the policy ran and the request went to the
    frontier. That includes a policy error and a frontier failure (a 502).
  - Both are NULL when no route policy ran: no policy for that tenant and provider, a policy with
    `stages: []`, a streamed request, a request for an allowlisted model, a request skipped for tools,
    and cache hits. A cache hit is traced through `cached_from` to the row that served it. A request
    held out for the frontier (`holdout` 1) has `policy_sha256` and a NULL `route_stage`.
- `requests.holdout` (integer, nullable): 1 for a request held out for the frontier, 0 for one routed
  through the policy, NULL outside the holdout comparison (see Measuring the saving under live
  routing).
  - The policy file is read only at startup. A changed file shows up as a new digest only after a
    restart.
- `requests.route_skip_reason` (text, nullable) says why a request went past its route policy
  untried, for example because it carried tools (see Routing through a policy).
- `requests.checks_json` (JSON text, nullable) keeps the named checks a route policy ran for the
  request, in order. `Ledger.get` returns it decoded as `checks`. There is no HTTP endpoint that reads
  one request; read the ledger row. Each entry has:
  - `check` (the check's name), `checked_model` (the model whose answer it read; null for a stage
    skipped before any model answered), `passed`, `cost_usd`, and `reason`, `error` and `status` when
    there are any. `model` is the stage's first model, as before.
  - `steps`: what each step decided, in order: `type` (`webhook` or `judge`), `passed`, `cost_usd`,
    and `reason`, `error` and `status` when there are any. Steps after the first rejection did not run
    and are not listed. A check cut off by the route deadline has no `steps`.
  - It is `[]` when the policy ran but no named check did (an agreement or unchecked stage), and NULL
    when no route policy ran, when its run raised an error, for cache hits, and for rows written
    before the column existed.
  - No request text, answer or patch is stored in it.
- Every outcome update is appended to the `outcome_events` table: `event_id`, `tenant`, `request_id`,
  `created_at`, `outcome_sequence`, `fields_json` and `previous_json`. `fields_json` holds the fields
  the PATCH set, plus `baseline_source` when it set a baseline. `previous_json` holds what those fields
  held before.
  - SQLite triggers make the table append-only: UPDATE and DELETE abort. It is not tamper-proof
    against someone with write access to the file.
  - Updates made before this version are not in the table. The `requests` table still holds only the
    latest values.
  - There is no HTTP endpoint for the history yet. Read it with `Ledger.outcome_events(tenant,
    request_id)` or SQL.
- `TINE_LEDGER_REQUIRED`: default off, and then an answer whose row cannot be written is served with
  `x-tine-ledger: unavailable`. With `1` or `true` such a request returns 503 with body
  `{"error":{"message":"The request could not be recorded in the ledger","type":"ledger_unavailable"}}`
  and the headers `x-tine-request-id` and `x-tine-ledger: unavailable`. This covers cache hits, routed
  answers (the cheaper calls were already paid), frontier answers including upstream error bodies, and
  streamed requests (the stream is closed before any byte is relayed). The 502 for an unreachable
  upstream is unchanged, since no answer is served. Shadow runs do not start for a request that got a
  503.

### Request limits

Three limits apply to `POST /v1/chat/completions` and `POST /v1/messages`. Each is checked after the
tenant token and before the verified cache, a policy, shadow mode or any provider call.

- `TINE_MAX_BODY_BYTES` is on by default at 10485760 bytes (10 MB). A body with a larger
  `Content-Length` gets 413 before it is read. A body without one is read until it passes the limit,
  then refused with 413. A body of exactly the limit is accepted. Set a whole number of at least 1.
  The limit does not cover `PATCH /v1/outcomes/{request_id}`, whose body is a few fixed fields.
- `TINE_DENIED_FIELDS` is off unless set. It is a JSON object like `TINE_ALLOWED_MODELS`, for example
  `{"openai": ["store"], "anthropic": ["mcp_servers", "container"]}`. A request that has a listed
  top-level field gets 400 naming the field, whatever its value, so `"store": false` is refused too.
  Only top-level keys are checked. A field listed for one provider is still forwarded to the other.
  An unknown provider, a string in place of an array or an empty name stops startup. Tools that the
  provider runs on its own side, such as `mcp_servers`, reach their targets from the provider's
  network, so an egress firewall around the proxy cannot block them. Deny the field here instead.
- `TINE_ANTHROPIC_BETA_ALLOWLIST` is off unless set. It is a comma-separated list of `anthropic-beta`
  values. The request's header is split on commas and trimmed, and if any value is not on the list the
  request gets 400 naming the refused values. A request without the header is unaffected. Unset or
  empty forwards every value, as before. It applies to `/v1/messages` only.

What is stored for a refused request: nothing. The 413 and 400 are returned before a ledger row,
cache entry or shadow count is written, and the response has no `x-tine-request-id`. The refusal
appears only in the server's access log, as the line uvicorn writes for every request (client address,
method, path, status). The body and headers are not logged.

These limits were built and tested with stand-in providers. No real customer has used them.

### One base URL per provider

Every model a policy names for a provider, each stage model and every judge model, is called at that
provider's one base URL (`TINE_OPENAI_BASE_URL` or `TINE_ANTHROPIC_BASE_URL`) with that provider's key.
The judge always goes through the Anthropic base URL. The shipped benchmark policies use open models
that api.openai.com and api.anthropic.com do not serve. Pointed at those APIs, each request makes 2 to 5
calls that fail, routing cost becomes unknown, and shadow mode stops after a few requests. Before
loading a policy, check that every model it names is served at the base URL.

To mix a frontier API with open models, point the base URL at one OpenAI-compatible gateway that you
run yourself and that serves all of them, such as vLLM or LiteLLM in your own network. A third-party
router would receive your prompts, which this project rules out. Failed stage calls are recorded with
their HTTP `status` beside the `error`, and a shadow stop names the most common failures (see Shadow
mode), so a model that is not served shows up as repeated HTTP errors from that model.

## Shadow mode

Shadow mode shows what a frozen policy ([POLICY.md](POLICY.md)) would have saved on a tenant's real
traffic, without changing anything the tenant sees. It is off unless configured.

- `TINE_SHADOW_POLICIES`: path to a JSON file mapping tenant to provider to policy, for example
  `{"acme": {"openai": { ...policy... }}}`. The policy's `frontier` must be that provider's configured
  frontier model, and its `tenant` must be the tenant it is listed under: a policy frozen for one
  tenant cannot be given to another, and a policy that names no tenant is refused. Name the tenant with
  `tine freeze-policy --tenant`; editing `tenant` by hand shows as an edit after freezing.
- `TINE_SHADOW_SAMPLE_RATE`: the share of eligible requests to shadow, 0 to 1.
- `TINE_SHADOW_MAX_COST_USD`: required when sampling. Shadow runs stop for a tenant once its known
  shadow spend reaches this amount. See "The spending cap" below for what it does and does not bound.
- `TINE_SHADOW_MAX_CONCURRENT`: shadow runs in flight at once, for the whole proxy. Default 8. A sampled
  request that arrives while all are busy is not shadowed; it is counted as `skipped_busy`.
- `TINE_SHADOW_MAX_CONCURRENT_PER_TENANT`: optional whole number, at least 1. Unset by default (no limit
  per tenant). It caps the shadow runs one tenant may have in flight, on top of the limit for the whole
  proxy, so one busy tenant cannot take every slot.
- `TINE_SHADOW_MAX_UNKNOWN_COST_CALLS` (default 20) and `TINE_SHADOW_MAX_UNKNOWN_COST_SHARE` (a number
  from 0 to 1, default 0.2): shadowing stops for a tenant when its shadow calls and checks of unknown
  cost are more than the count **and** more than the share of all its shadow calls and checks. A share
  of 0 gives the older rule, on the count alone. Three CI flakes in 20 requests (40 calls) no longer
  stop shadowing.
- `TINE_SHADOW_MAX_CALLS_PER_MINUTE`: optional whole number, at least 1. Unset by default (no cap). It
  caps the shadow provider calls (stage models and judges, not webhooks) one tenant may start in any
  60 seconds. See "Rate limits" below.
- `TINE_ANSWER_PATTERN`: optional regular expression; the comparable answer of a response is its
  last match (first group if there is one). A policy's own `answer_pattern`, written by
  `tine freeze-policy --answer-pattern`, wins; this setting is the fallback for policies without one.
  To compare whole replies on purpose, set the pattern `(?s).+`. A shadow policy with agreement stages
  and no reader still starts, but logs a WARNING and `GET /v1/shadow` shows it: whole reply texts are
  then compared, and free text almost never agrees.
- A policy's `answer_compare` says how two answers are compared once they are read: `exact` (the
  default, also when the field is absent) or `math`, by value after dropping LaTeX notation such as
  `\%` and `\dfrac` ([POLICY.md](POLICY.md), Comparing answers). It is used both for agreement
  between cheaper models and for comparing an accepted answer with the frontier's (`matches_frontier`
  and the drift check). There is no environment setting for it: it is part of what was frozen.

**One tenant token per task type.** The proxy does not read `task_type`: every request on a tenant's
token, for that provider, goes through the tenant's one policy. With one token for mixed traffic, a
multiple-choice pattern can read a trailing "(c)" in a free-form draft as an answer, and drafts can be
accepted. Give each task type its own tenant token and its own policy, check the `task_type` of each
loaded policy in the startup log, and read the acceptance share of each token in shadow mode before
routing any of them. A token whose acceptance is far from its policy's estimate is carrying traffic
the policy was not certified on.

At startup the proxy logs one INFO line (logger `tine.proxy`) per loaded shadow and route policy:
tenant, provider, `task_type`, number of cheaper stages, the answer reader and where it comes from,
how answers are compared (`answers compared exact` or `math`), and the state of the freeze digest ([POLICY.md](POLICY.md)). A WARNING line is logged for a policy
with agreement stages and no reader, and for a file edited after freezing. `tine serve` attaches an
INFO handler to the `tine` logger only (format `LEVEL:     name: message`).

A request is eligible when it was answered by the frontier model, without streaming, without a cache
hit and without error. The caller receives the frontier's response first; the policy's cheaper stages
then run in the background against the same provider with the same request, and one row is recorded:
whether the policy would have accepted a cheaper answer, whether that answer equals the frontier's, and
what each call cost. A failure anywhere in a shadow run is contained and changes nothing for the caller.
Shadow runs, and every call a policy makes, use their own connections, separate from the ones that
carry frontier requests. A slow cheap model or webhook can fill only those.

`GET /v1/shadow` returns the tenant's report: requests shadowed, the share the policy would have
answered with cheaper models, the share of those whose answer matched the frontier's, the frontier's
cost against the policy's projected cost (every cheaper call, plus the frontier where nothing was
accepted), and what shadow mode itself has spent. `shadow_spend_usd` is a subtotal of known costs. It
is zero, not unknown, when no cost is known, so read it with `unknown_cost_shadow_calls`, the number of
calls and checks whose cost is not known. `sampling` says whether new shadow runs can start for the
tenant: `on`, or `stopped` with the reason.

More fields of `GET /v1/shadow`:

- `policies`: provider to `{task_type, cheaper_stages, answer_reader: {answer_pattern, source},
  answer_compare, freeze_digest, warning?}`. `source` is `policy file`, `TINE_ANSWER_PATTERN`, or
  `none: whole reply texts are compared`. `answer_compare` is `exact` or `math`. It is `{}` for a
  tenant without a shadow policy.
- `sent_to_frontier_for_tools`: sampled requests that defined tools and went to the frontier untried
  (see Routing through a policy). They are counted inside `shadowed_requests`.
- `check_step_rejections`: check name to step type to the number of times a step of that type turned
  an answer down, for example `{"tests+judge": {"webhook": 3, "judge": 5}}`. A step that failed (an
  error or timeout) counts, because a check fails closed. A stage skipped before any model answered
  has no steps and is counted in `check_rejection_reasons` instead. Runs stored before steps were kept
  are not counted. `{}` when there are none.
- Each check entry in `shadow_runs.calls_json` keeps `checked_model` (the model whose answer it read)
  and `steps`, as in the ledger's `checks_json` (see Audit trail). Its `model` stays
  `check:<name>`.
- `check_rejection_reasons`: reason (`request_too_long`, `response_too_long`) to the number of shadow
  check records with it; `{}` when there are none. A shadowed request whose stages were all skipped
  still counts in `shadowed_requests`, at shadow cost $0 and not accepted (the frontier would answer).
  The skipped check is one entry in `shadow_calls` at cost 0.
- Coverage, kept in a ledger table `shadow_counts`:
  - `sampled_requests`: sampled requests a shadow run could have taken (answered by the frontier without
    streaming, no fallback, no error), counted once their run ended or was skipped. It equals
    `shadowed_requests` + `skipped_busy` + `skipped_stopped` + `failed_shadow_runs` +
    `skipped_rate_limited` + `skipped_rate_cap` + `rate_limited_runs`.
  - `skipped_busy`: sampled but not run because a concurrency limit was full, or because the spending
    cap could not reserve room. That happens while runs in flight hold the room, or before the tenant's
    first run has finished, when one run at a time is let through.
  - `skipped_stopped`: sampled after shadowing had stopped (cap reached, or the unknown-cost stop).
  - `failed_shadow_runs`: the run raised, or its row could not be written.
  - `skipped_rate_limited`: sampled while shadow mode was paused on a provider the run would call (see
    Rate limits).
  - `skipped_rate_cap`: sampled when the run could have passed `TINE_SHADOW_MAX_CALLS_PER_MINUTE`.
  - `rate_limited_runs`: runs in which a provider refused a call with 429 or 529, or a call was not
    sent because shadow mode had paused. Their rows are kept, so their spend is in `shadow_spend_usd`
    and `shadow_calls`, but they are not in `shadowed_requests`, the projection or the drift check.
  - `streamed_not_shadowed`: sampled requests that streamed. Shadow mode never runs them; they are
    counted only when they would otherwise have been shadowed.
  - `shadow_coverage`: `shadowed_requests` / `sampled_requests`, or null when nothing was sampled.
    Coverage below 1 means busy periods count for less than they should in the projection.
  - The counts start when a proxy with this version first opens the ledger, so an older ledger's
    earlier rows look fully covered. A read-only copy of a ledger without the table gives null for all
    nine fields. Other traffic that is never eligible (cache hits, an allowlisted cheaper model, frontier
    errors, non-JSON frontier replies) is not counted.
- Drift:
  - `accepted_differing_from_frontier` and `accepted_differing_from_frontier_share`: accepted cheaper
    answers whose answer differs from the frontier's, as a count and as a share of the answers
    compared. Answers accepted by a named check (code) are not compared and are left out.
  - `drift_requests` and `drift_share`: the shadowed requests whose answer was compared with the
    frontier's or sent to the frontier (all shadowed requests minus accepted answers that were not
    compared), and the differing answers as a share of them. Epsilon bounds the loss over all tasks,
    and a request sent to the frontier loses nothing, so this is the share checked against epsilon.
    The share among accepted answers alone is larger by about one over the acceptance rate.
  - `drift_epsilon`: the strictest `guarantee.epsilon` among the tenant's shadow policies that have
    stages. `drift_min_compared`: 30.
  - `drift_check`: `warning`; `within the limit: ...`; `not checked yet: N accepted answers compared
    with the frontier's, 30 needed`; or `not checked: the policy's quality limit was not given`.
  - `drift_warning`: null unless `drift_share` is above epsilon on at least 30 compared answers. It
    then reads: "Cannot confirm the certification on this traffic: D of C accepted cheaper answers
    differ from the frontier model's answer, S% of the R shadowed requests whose answer was compared
    with the frontier's or sent to the frontier, more than the policy's quality limit of E%. A
    different answer is not always a wrong one; grade a sample of these requests before relying on the
    policy or its projected saving."

The drift warning is a reason to grade, not a measured loss. A different answer can be the frontier's
mistake. On the published benchmark replay 58 of 1,099 accepted answers (5.28%) differ from the
frontier's, 4.10% of the 1,414 shadowed requests, against an epsilon of 2%, so the warning shows
there too, although the graded loss was 0.99%. The
planned next step is to record the differing rate at certification time and compare against that.
The check uses the tenant's strictest epsilon across providers, not each policy's own, and counts rows
from older policy files too.

### Rate limits

Rate limits belong to the provider organisation, so shadow calls and the tenant's real calls share
them. In a live pilot on an account with modest limits, shadow calls running beside real traffic led
Anthropic to answer 429 to 19-38% of the real frontier requests in the shadow runs, 25-58% in the routed
runs and 65% in the server test (`results/2026-11-17-live-pilot-more`). Shadow mode now holds back:

- A provider reply of 429 (rate limit) or 529 (overloaded) pauses shadow mode on that provider. This
  holds for a shadow call, a judge call made by a shadow run, a routed request's cheaper call and the
  real frontier request. Webhooks are not provider calls and do not pause anything.
- The pause is per provider, for every tenant. The proxy calls each provider with one key, so all
  tenants share that account's limits.
- While a pause lasts, no new shadow run starts on that provider. A run whose policy has a judge step
  also waits for a pause on Anthropic. Sampled requests are counted as `skipped_rate_limited`.
- The pause lasts as long as the reply's `retry-after` header asks, in seconds (at least 1, at most an
  hour). An HTTP date in that header is not read. Without one, the pause is 30 seconds, doubled for each
  pause in a row, up to 10 minutes. A shadow run that ends with no refused call resets the doubling.
- A refused call is never retried. Real requests are passed through as before, with the provider's
  status and `retry-after`.
- A run in flight sends no more provider calls once its provider is paused. Such a run, and any run in
  which a provider refused a call, is stored with `skip_reason` "rate_limited: ..." and counted in
  `rate_limited_runs`. It is left out of the projection, because it does not show what the policy
  would do.
- A refused call, or one not sent, is stored at cost $0: a provider does not bill a request it refused.
  So refusals do not count toward the unknown-cost stop.
- `TINE_SHADOW_MAX_CALLS_PER_MINUTE` caps a tenant's shadow provider calls in any 60 seconds. A run
  reserves the most calls it can make (each stage model once, plus each judge step) when it starts,
  and gives back the ones it did not make when it ends. A sampled request whose run would pass the cap
  is skipped and counted as `skipped_rate_cap`. The proxy refuses to start when one run of a shadow
  policy can make more calls than the cap.
- The pauses and the call count are kept in memory, in each proxy process. A restart clears them.
- `GET /v1/shadow` also returns `shadow_paused` (provider to null, or to `{paused_until, seconds_left,
  reason}`, for each provider the tenant's shadow policies call), `shadow_max_calls_per_minute` and
  `shadow_calls_last_minute` (null without a cap). `sampling` still says `on` during a pause.

Status: built and tested with stand-in providers that answer 429 and 529, with and without
`retry-after`, on a test clock. It met a real provider's limit once
(`results/2026-11-17-concurrency-and-long-shadow`): in a test added after that run's plan, the provider
refused 9 of 69 real requests without the pause and none of 60 with it, and 55 of those 60 were not
shadowed. The planned test did not meet its pass marks: no real request was refused in it, so it did
not exercise the pause. The pause was not seen to end, and a `retry-after` header and a 529 were not met.
It lowers the load shadow mode adds; it cannot stop real traffic alone from reaching the limit.

The same report as a page to send to a customer:
`python scripts/savings_report.py --ledger LEDGER --tenant TENANT --policy POLICY.json --output report.html`
([REPORT.md](REPORT.md)). `--note` adds one line under the headline. `POLICY.json` can be the
`TINE_SHADOW_POLICIES` or `TINE_ROUTE_POLICIES` file itself: the script takes the tenant's policy from it.

### The spending cap

- The cap counts known costs. At startup the proxy refuses to sample unless every stage model of every
  shadow policy has an input and output price in `TINE_MODEL_PRICES`, and every judge step those
  policies use has an `anthropic:<model>` price.
- A call can still have an unknown cost: it failed or timed out, the reply carried no usage, or a
  webhook reported a cost that is not a usable amount. Such calls cannot be counted in dollars. They
  are counted by number, and shadowing stops for the tenant once there are more than
  `TINE_SHADOW_MAX_UNKNOWN_COST_CALLS` and they are more than `TINE_SHADOW_MAX_UNKNOWN_COST_SHARE` of
  its shadow calls and checks. What those calls cost is not bounded in dollars by anything here.
- Failed shadow and route calls record `status` (the HTTP code) beside `error`. A named check that
  fails closed records the step and the error type (for example `error: "webhook HTTPStatusError"`)
  and `status`, so webhook and judge failures count in `failed_shadow_calls`.
- The stop reason in `sampling` reads: "stopped: too many shadow calls of unknown cost: N of M calls
  and checks (P%), more than <count> and more than <share> of them. Most common: <model or
  check:name>: HTTP <status> (<times>); ...". Up to three causes are named. Without an HTTP status the
  cause is the error type (for example `DeadlineExceeded`); with no error it is "no usage or no price in
  the reply". The spending-cap stop reads "stopped: the spending cap is reached".

To resume after an unknown-cost stop. The stop is rebuilt from the tenant's stored shadow rows at
every start, so a restart alone does not resume it, and nothing new is shadowed while it lasts.

1. Fix the cause the stop reason names: every stage model must be served at the provider's base URL,
   the CI webhook must be healthy, and every stage and judge model needs a price.
2. Then either raise `TINE_SHADOW_MAX_UNKNOWN_COST_CALLS` and/or `TINE_SHADOW_MAX_UNKNOWN_COST_SHARE`
   and restart, or back up the ledger, delete that tenant's rows from its `shadow_runs` table and
   restart. Deleted rows leave the report, and their spend leaves the spending cap's count.

There is no reset endpoint.
- When a check has several steps and one step's cost is unknown, the cost of the other steps is still
  counted.
- A run's cost is known only when it ends. A run in flight therefore reserves room under the cap: it is
  assumed to cost what the tenant's most expensive run so far did. Until one run has finished, one run
  at a time is let through. The run that crosses the cap still completes. Spend can pass the cap by
  about one run, and by more only when runs in flight cost more than any run before them.
- The count is kept in memory in each proxy process. It is read from the ledger once per tenant and
  then updated as runs end, even if a run's row could not be written. Two proxy processes on one ledger
  each keep their own count, so together they can spend up to the cap each. Run one process per ledger
  if the cap must hold exactly.
- A run cut off when the proxy shuts down is not recorded and not counted.

Limits to state with any number from it:

- **Shadow calls are real calls on the tenant's provider account.** They cost money and they send the
  tenant's prompt to the cheaper models on that account. Nothing goes to a third party.
- **Equal answers are not proof of correctness.** The comparison is meaningful for exact answers (a
  label, a number, a choice, usually with an answer pattern). Free-form answers rarely match word for
  word, so such traffic shows low acceptance here. The quality limit comes from the policy's own
  certification on graded tasks.
- A stage with a named check (tests, a judge) is shadowed only if that check is configured (see Named
  checks below); otherwise the policy is refused at startup. For such stages the accepted answer is not
  compared with the frontier's, because two right patches rarely match word for word.
- Projected savings use only requests whose costs are all known, so every stage model needs a price in
  `TINE_MODEL_PRICES`. Streaming requests are never shadowed; sampled ones are counted in
  `streamed_not_shadowed`. The streamed share of traffic caps any saving. A streamed request's own
  cost is recorded in the ledger when its stream carries usage (see above); that does not change these
  counts.
- Shadow results store no prompt or response text; the request record already holds the tenant's request
  and the frontier's response.

## Routing through a policy

With a route policy the proxy serves answers from a frozen policy. It is off unless configured, and a
tenant's provider has either a shadow policy or a route policy, never both. Run shadow mode first.

- `TINE_ROUTE_POLICIES`: path to a JSON file, tenant to provider to policy, as for shadow mode.
- `TINE_ANSWER_PATTERN`: how a comparable answer is read from a response, for agreement stages, when
  the policy has no `answer_pattern` of its own. The proxy refuses to start when a route policy has an
  agreement stage and no reader at all: "route_policies: tenant '<t>''s <provider> policy has agreement
  stages but no answer reader, so whole reply texts would be compared. Freeze it with 'tine
  freeze-policy --answer-pattern REGEX', or set TINE_ANSWER_PATTERN". Policies with only named checks,
  or with `none`, need no reader. Agreement compares the answers it reads as the policy's
  `answer_compare` says, as in shadow mode. With `math` the caller is served the first cheaper model's
  reply as it was written, not the rewritten answer.
- `TINE_ROUTE_ESTIMATE_BASELINE`: `true` to estimate the frontier baseline for a routed answer (below).
- `TINE_ROUTE_DEADLINE_SECONDS`: how long the cheaper stages may take together for one request. Default:
  30 seconds, or `TINE_TIMEOUT_SECONDS` if that is shorter. When it runs out the request goes to the
  frontier.
- `TINE_AUDIT_RATE`: the share of requests held out for the frontier, between 0 and 1. Default 0, off
  (see Measuring the saving under live routing).

For a non-streaming request that would have gone to the frontier, the proxy tries the policy's stages in
order, with the same request and the model swapped. A stage's models are called together. If the
stage's check passes, the first member's response is returned as it came from the provider, with
`x-tine-model` naming it and `x-tine-route: stage-N`. If no stage is accepted, or anything in the policy
path fails, the request goes to the frontier exactly as without a policy (`x-tine-route: frontier`).
Streaming requests and requests for an allowlisted non-frontier model are not routed.

Replies that are never served from a stage, whatever its check:

- A 200 reply with no answer in it: no text and no tool call, an error body, a content-filtered reply
  or a refusal. It is a provider failure, not an answer. What it cost is kept as routing cost.
- A reply that holds `NaN` or `Infinity`, which JSON cannot carry.

For an agreement stage, a reply that calls a tool or did not finish normally (cut off by the token
limit, for example) has no answer to compare, so the stage does not pass. The policy was certified on
final answers, not on tool use.

**Requests with tools.** A request whose body has a non-empty `tools` or `functions` list skips any
policy that has an agreement stage and goes straight to the frontier. The whole chain is skipped, not
only its agreement stages, because a chain with a stage removed is not the chain that was certified.
The response carries `x-tine-route: frontier` and `x-tine-route-skipped: tools`. The ledger row has
`routing_cost_usd` 0.0 and `route_skip_reason` "tools: the request carries tool definitions, and
agreement cannot compare tool calls". In shadow mode a sampled tool request makes no cheaper calls; it
is stored as a not-accepted run at the frontier's cost (`shadow_runs.skip_reason`) and counted in
`sent_to_frontier_for_tools`. Nothing is stored once shadowing has stopped. Policies whose stages all
use named checks still run on tool requests, and a tool-use turn cannot pass a code check, so such a
request still pays the cheap call and the check before it goes to the frontier.

**Requests too long for the judge.** A stage whose check has a judge step is skipped before its cheap
models are called when the request's user text is over 12,000 characters (see Named checks). No cheap
call and no webhook run are paid, and the request goes to the next stage or the frontier. A skipped
stage adds nothing to routing cost; when every stage is skipped the row shows `routing_cost_usd` 0.0
and `x-tine-route: frontier`. The reason is recorded only in shadow runs, not in the ledger or a
response header.

The deadline bounds what routing adds to a request. Without it, a stalled cheap model or webhook would
hold every routed request for each stage's own timeout in turn. A call or check cut off at the deadline
is recorded with an unknown cost, because the provider may still charge for it. The default deadline
is short for code: a stage whose tests take longer than it never passes, and each such request then
pays the cheap call and the frontier. Set the deadline to cover the test run.

A routed answer that is later confirmed through the outcome endpoint is served from the verified cache
when the same request comes again. The ledger row of the cache hit names the model whose answer it is.

What the ledger records for a routed request:

- `cost_usd`: the returned answer's cost. `routing_cost_usd`: every other call the policy made for the
  request (second opinions, attempts that were escalated, judge calls, and whatever a webhook reports).
  It is left unknown if any of those costs is unknown, for example a model with no configured price or
  a cheaper call that failed before its usage was known.
- `baseline_cost_usd`: what the frontier would have cost is not observed when it was not called. It is
  left unknown unless `TINE_ROUTE_ESTIMATE_BASELINE` is on, in which case it is **an estimate**: the
  returned answer's token counts at the frontier's prices. A frontier answer could be longer or
  shorter, so savings computed from it are approximate. Supply a measured baseline through the outcome
  endpoint when one exists. When the request escalates, the baseline is the frontier's actual cost.
- An estimated baseline is marked as one. The row has `baseline_estimated` set and the response carries
  `x-tine-baseline: estimated`. `GET /v1/savings` keeps such requests out of `verified_savings_usd` and
  `savings_fraction`. It reports them apart as `estimated_baseline_requests` and
  `estimated_baseline_savings_usd`, with `estimated_baseline_scope` saying what the figure rests on. A
  baseline supplied later through the outcome endpoint replaces the estimate. It is then recorded as
  `caller` and reported with the caller-supplied figures, never as measured (see Where a baseline
  comes from).
- `policy_sha256` and `route_stage` name the policy file and the stage behind the row, and
  `checks_json` which answers the named checks read and what each step decided (see Audit trail).
- If the frontier cannot be reached after the policy escalated, the 502 row still records what the
  cheaper attempts and checks cost.

The quality limit of a route policy is the one certified when it was frozen, on the tasks it was
frozen from. The proxy does not check that live traffic resembles those tasks; shadow mode's drift
warning is the only signal. Outcomes still have to be reported through the outcome endpoint for
savings to count as verified. Under live routing the saving itself is measured only with a frontier
holdout (`TINE_AUDIT_RATE`; see Measuring the saving under live routing).

Limits of routing and shadow mode that are not fixed in code:

- **Answers are not read the way they were when the policy was certified.** The multiple-choice policy
  was certified on answers read by `scripts/import_xroutebench.py`: the last letter in brackets, upper
  case; or the last number, with commas and trailing zeros removed; or the last `\boxed{}`. The proxy
  reads an answer with one regular expression and compares the raw match. A policy frozen with
  `--answer-pattern` names its pattern, but the pattern is not part of what was certified, and the two
  shipped benchmark policies name none. Two effects follow. With a pattern narrower than the certified
  reader (one that accepts `(A)` but skips `(b)`), the proxy can accept answers the certified check
  would reject. With the closest pattern, answers that differ only in form (`(b)` and `(B)`, `1,000.50`
  and `1000.5`) do not agree, the request pays every cheap call and then the frontier, and live savings
  fall below the stored estimate. Compare the acceptance share in shadow mode with the policy's
  estimate before routing. The replay in `results/2026-11-16-shadow-replay` does not test this: its
  stand-in provider returns answers that were already read.
- **No memory of failures between requests.** While a cheap model or check is down, every routed
  request still tries it and waits up to the deadline. There is no circuit breaker. The members of a
  stage are awaited together, so one slow member holds the stage even when another has already failed.
- **A policy file is not signed.** The proxy checks the guarantee's numbers, that the file names the
  tenant it is configured for and that no stage repeats a model ([POLICY.md](POLICY.md)). A file
  frozen with a digest that was edited afterwards still loads; the startup log, `GET /v1/shadow` and
  the report page flag it. Anyone who can edit the file can also write a new digest, and a file with
  no digest cannot be checked at all. `task_type` is not read per request: every routed request of a
  tenant and provider goes through that tenant's one policy for the provider (see One tenant token per
  task type).
- **Policy files are read only at startup.** Editing or emptying one changes nothing until a restart.
  [DEPLOY.md](DEPLOY.md) has "Turn routing off".

## Named checks

A policy stage for code names the check it was certified with, for example `tests+judge:sonnet`.
`TINE_CHECKS` is a path to a JSON file that tells the proxy how to run each name: a list of steps, run in
order, all of which must pass.

```json
{"tests+judge:sonnet": [
  {"type": "webhook", "url": "https://ci.internal.example/tine-check", "bearer_env": "CI_TOKEN"},
  {"type": "judge", "model": "claude-sonnet-5-5", "effort": "medium"}
]}
```

- `webhook`: the proxy POSTs `{"tenant", "provider", "model", "request_id", "mode", "request",
  "response"}` to a URL the operator controls. `request_id` is the proxy's id for the request (the
  `x-tine-request-id` header and the ledger row) and `mode` is `"shadow"` or `"route"`, so a run on
  the customer's side can be matched with the proxy's records. It reads back
  `{"passed": true|false, "cost_usd": optional}`. This is how a tenant's own tests are run: beside
  their repository, on their infrastructure. **The proxy never executes model-written code.** `bearer_env` names an environment variable holding a token to send; the
  token is never written to the configuration. `timeout_seconds` defaults to 60 and may be at most 600.
  A `cost_usd` that is not a usable amount (negative, `NaN`, infinite, or above one million dollars)
  makes that check's cost unknown. It is never added to a total.
  - **The webhook contract.** It should run the customer's own tests on the patch and report whether
    they pass: the tests that exist in the repository, plus any the change adds. It must not be a
    grader that knows the right answer. The pilot brief's stand-in webhook agreed with the grade on 819
    of 819 answers, and the chain then lost nothing; with a realistic webhook that reports only
    regressions, 19 of 99 accepted patches were wrong (1.0 to 1.5% of tasks lost). The measured chains
    assume tests of the second kind.
- `bearer_env` rules. The proxy refuses to start when any of these is broken:
  - It must name a variable of the webhook's own. A name that starts with `TINE_`, or equals
    `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, is refused, ignoring case: "checks: Check 'x': bearer_env
    'TINE_TENANT_KEYS' names a setting of the proxy or a provider key...".
  - A webhook step with `bearer_env` must use an `https://` url. There is no exception for loopback:
    http to 127.0.0.1 or localhost with a token is refused too. A webhook with no `bearer_env` may
    still use http.
  - One token variable may be sent by the webhooks of only one tenant. The same `bearer_env` in two
    tenants' `TINE_TENANT_CHECKS` entries is refused, and so is one in a shared check that one tenant's
    policies use and also in another tenant's own entry. Give each tenant its own variable, as in the
    example below.
- **Check files are operator-only configuration.** Whoever can edit `TINE_CHECKS` or
  `TINE_TENANT_CHECKS` chooses which URLs receive tenants' requests and which environment variable is
  sent as a token. The reserved names above cover only the proxy's own settings and the provider keys;
  a check file can still name any other variable in the proxy's environment, such as a cloud
  credential. Keep only webhook tokens and the proxy's own settings in that environment, and never let
  a tenant edit a check file.
- `judge`: the proxy asks a Claude model through its configured Anthropic provider whether the response
  resolves the request, with the instructions the measured judges used. `threshold` accepts on the
  judge's probability instead of its verdict. The judge's call is priced from `TINE_MODEL_PRICES`
  (`anthropic:<model>`) and counted as routing cost.

`TINE_CHECKS` holds checks every tenant may use. `TINE_TENANT_CHECKS` is a path to a second JSON file,
tenant to check name to steps, for checks that belong to one tenant:

```json
{"acme":   {"tests+judge:sonnet": [{"type": "webhook", "url": "https://ci.acme.example/check", "bearer_env": "ACME_CI_TOKEN"}]},
 "globex": {"tests+judge:sonnet": [{"type": "webhook", "url": "https://ci.globex.example/check", "bearer_env": "GLOBEX_CI_TOKEN"}]}}
```

A tenant's own check wins over a shared check of the same name, and a tenant cannot use another
tenant's check. Policies frozen the same way carry the same check name, so this is how two tenants get
different webhooks. The proxy refuses to start when a shared check with a webhook step is named by the
policies of more than one tenant: that webhook would receive one tenant's request, and its token, on
behalf of another. If one endpoint really serves several tenants, list it under each of them.

The judge step and what it reads:

- The judge reads at most 12,000 characters of the request's user text and 20,000 of the response. A
  longer text is not cut. When the user text is longer, any check with a judge step cannot pass, so
  the stage is skipped before its cheap models are called: no cheap call and no webhook run are paid,
  and the request moves to the next stage or the frontier. The check is recorded as not passed with
  reason `request_too_long`. An answer over 20,000 characters is rejected after the call with reason
  `response_too_long`. Checks with no judge step are unaffected. System, assistant and tool messages
  are not shown to the judge.
- **Request shape.** Agents should put repository context in `system`, which the judge does not read,
  and keep the user turn to the issue or task, at most 12,000 characters. That is the shape the judge
  was measured on: it read the issue and the patch. In the simulated coding pilot, 252 of 274 prompts
  had repository context in the user turn, so the judge never ran on them and the chain saved nothing.
- The judge is not asked about a reply that calls a tool or was cut off by the token limit. It reads
  only the reply's text, so it would rule on the words while the proxy served a tool call it never saw.
  Such a reply is rejected without a judge call, with reason `not_a_finished_reply`, and the request
  moves on.
- The text the judge reads is not trusted. The tags that frame it (`<issue>`, `<patch>` and their
  closing forms) are made inert wherever they occur inside it, so text cannot close its frame and
  continue as if it were instructions.
- A reply from the judge that holds verdict objects that disagree is not a verdict, and rejects.
- The judge is told that the tests passed only when a webhook step passed earlier in the same check. In a check with no webhook step it is told the patch has not been applied or tested. That
  judge is not the one that was measured.

Every step fails closed: an error, a timeout, a malformed reply or a refusal rejects the answer, so the
request moves on. Limits:

- A configured check must do what the measured one did, or the policy's guarantee does not describe
  what runs. The measured judges read a GitHub issue and a unified diff
  (`results/2026-11-14-api-judge`); here the judge reads the request's user text and the response
  text. Its measured catch rate carries over only to traffic of that shape, and it was measured on
  public repositories the judge may have seen.
- A webhook receives the tenant's request and a model's response. Point it only at the tenant's own
  infrastructure.
- The judge's instructions are the measured ones. They do not tell the judge that the text may contain
  instructions to ignore. Simple injected approvals in a patch did not move the judges offline
  (`results/2026-11-16-judge-probes`); stronger attacks were not tried, and none through the proxy.
- This path has been tested with stand-in providers only. It has not served real traffic.
