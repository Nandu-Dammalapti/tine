# Local proxy and ledger

The development proxy exposes native OpenAI `POST /v1/chat/completions` and Anthropic `POST /v1/messages` routes, including streaming passthrough. It uses an authenticated tenant mapping and defaults every request to that provider's configured frontier model. The offline experiment does not automatically enable cheaper live routing.

Use the activated Python environment from the README. Configure these variables through your shell's secure configuration or environment settings:

| Variable | Meaning |
| --- | --- |
| `TINE_TENANT_KEYS` | JSON object mapping incoming proxy tokens to tenant IDs |
| `TINE_OPENAI_API_KEY` / `TINE_ANTHROPIC_API_KEY` | Provider credentials; configure only the providers you use |
| `TINE_OPENAI_FRONTIER_MODEL` / `TINE_ANTHROPIC_FRONTIER_MODEL` | Exact frontier model name for each provider |
| `TINE_LEDGER_PATH` | SQLite file, default `artifacts/ledger.sqlite3` |
| `TINE_OPENAI_BASE_URL` / `TINE_ANTHROPIC_BASE_URL` | Optional API-compatible upstreams; defaults are official HTTPS endpoints |
| `TINE_TIMEOUT_SECONDS` | Upstream timeout, default 30 |
| `TINE_MODEL_PRICES` | Optional explicit USD-per-million token rates keyed by `provider:model` |
| `TINE_VERIFIED_CACHE` | `true` by default; cache requires an explicit successful outcome |

Never commit credentials or paste provider keys into notebook cells. The app refuses startup without configured tenant keys. Clients use their **proxy tenant token**, not the provider key, in `Authorization: Bearer ...`; Anthropic clients can use `x-api-key` for that proxy token. Outbound requests use the server's provider credential. A caller-supplied tenant header cannot select a tenant.

Start locally:

```bash
python -m tine serve --host 127.0.0.1 --port 8000
```

`GET /health` reports frontier-default routing and SQLite development storage. Provider routes return a Tine request ID in `x-tine-request-id`. After an objective check completes, authenticate as the same tenant and update it with `PATCH /v1/outcomes/{request_id}` and JSON `{"success": true}` or `{"success": false}`. Optional `cost_usd`, `routing_cost_usd`, and `baseline_cost_usd` must come from actual accounting; they are not guessed. `GET /v1/savings` returns that tenant's ledger summary.

Record the returned answer's generation cost in `cost_usd` and the additional routing cost in `routing_cost_usd`. Routing cost includes earlier attempts that escalated, judges, tests, embeddings, and allocated customer-hosted compute. Include each expense once. If an older caller included these expenses in `cost_usd`, split that amount before supplying the routing component. `baseline_cost_usd` is the comparable always-frontier cost.

For example, a verified answer costing $0.25 plus $0.90 in routing against a $1.00 baseline has **-$0.15 net savings**. Send `{"success": true, "cost_usd": 0.25, "routing_cost_usd": 0.90, "baseline_cost_usd": 1.00}`. Omitting a cost on later updates preserves it; explicit `null` marks it unknown again. Zero must be an explicit accounting result.

Automatic proxy records leave routing cost unknown, including cache hits: no provider call does not imply free checking or infrastructure. Cache hits do not copy the source request's routing cost. Older SQLite ledgers gain a nullable routing-cost column; existing rows stay unknown until costs are supplied. Restart all writers together when upgrading. Cache eligibility still depends on the verified outcome, independently of whether costs have arrived.

Net savings require known answer, routing, and baseline costs for a verified successful request. `verified_cost_usd` now includes both cost components, exposed separately as `verified_answer_cost_usd` and `verified_routing_cost_usd`. `verified_savings_usd` and `savings_fraction` are null when no comparable request exists. `unknown_cost_requests` now counts requests missing either component; separate answer/routing counts identify the gaps. Known cost totals sum measured components across successes, failures, and pending requests, so they are incomplete while any component is unknown. Cost per verified success is null until every request has both costs. Unrepresentable numeric aggregates are also null.

Unknown outcomes and costs remain unknown. Cached responses are scoped by tenant, provider, upstream endpoint, model, exact request contents, and forwarded Anthropic version/beta headers. They become eligible after success is explicitly confirmed. After upgrading, responses recorded with the older cache keys require a fresh upstream request and successful outcome before cache reuse, because those keys lack upstream context. Streaming responses are not cached. The ledger stores prompts/responses; use only data you intend to retain in the configured local database.

For OpenAI-compatible usage, omitted optional cache details retain the zero-cache default. Explicitly null or malformed `prompt_tokens_details` instead leave cached tokens, estimated cost, and the automatic frontier baseline unknown, even if input/output totals are valid. The provider response still passes through unchanged; a later measured cost can be supplied through the outcome endpoint.

Late outcome reversals or newer resolved failures invalidate prior cached successes. Resolved outcomes use a transactionally assigned database sequence, so equal or backward-moving wall-clock timestamps cannot revive stale success evidence. Existing ledgers are upgraded automatically; legacy entries retain their prior timestamp order until new sequenced evidence arrives. Restart all proxy writers together when upgrading so every writer uses this ordering. If local ledger recording fails, a successful provider response is still returned with `x-tine-ledger: unavailable`. Net savings reports cover comparable, verified successful requests only; they are not billing amounts or proof of end-to-end savings. Failed and pending requests with known costs contribute to total spend, including routing costs, and to the cost-per-verified-success metric.

`TINE_ALLOWED_MODELS` can explicitly allow client-requested development alternatives, with a single frontier retry on an upstream failure before a response is emitted. Supply a JSON object of provider names and model-name arrays, for example `{"openai": ["EXACT_MODEL_NAME"]}`. Names match exactly. A string in place of an array, an unknown provider key, or an empty model name causes startup to fail before a ledger is opened. This switch is not a calibrated router. Failure after streaming starts cannot safely be retried without duplicating output. Keep it unset for frontier-only operation.

This is a development implementation: SQLite is not the planned production Postgres backend. Tests use mock upstreams, not paid API calls. Do not deploy it as a production guarantee or host it on Colab. No live provider credentials were requested or validated during this work.

## Shadow mode

Shadow mode shows what a frozen policy ([POLICY.md](POLICY.md)) would have saved on a tenant's real
traffic, without changing anything the tenant sees. It is off unless configured.

- `TINE_SHADOW_POLICIES`: path to a JSON file mapping tenant to provider to policy, for example
  `{"acme": {"openai": { ...policy... }}}`. The policy's `frontier` must be that provider's configured
  frontier model, and a policy frozen for one tenant cannot be given to another.
- `TINE_SHADOW_SAMPLE_RATE`: the share of eligible requests to shadow, 0 to 1.
- `TINE_SHADOW_MAX_COST_USD`: required when sampling. Shadow calls stop for a tenant once its recorded
  shadow spend reaches this amount; the run in progress when the cap is crossed still completes.
- `TINE_ANSWER_PATTERN`: optional regular expression; the comparable answer of a response is its
  last match (first group if there is one). Without it, whole response texts are compared.

A request is eligible when it was answered by the frontier model, without streaming, without a cache
hit and without error. The caller receives the frontier's response first; the policy's cheaper stages
then run in the background against the same provider with the same request, and one row is recorded:
whether the policy would have accepted a cheaper answer, whether that answer equals the frontier's, and
what each call cost. A failure anywhere in a shadow run is contained and changes nothing for the caller.

`GET /v1/shadow` returns the tenant's report: requests shadowed, the share the policy would have
answered with cheaper models, the share of those whose answer matched the frontier's, the frontier's
cost against the policy's projected cost (every cheaper call, plus the frontier where nothing was
accepted), and what shadow mode itself has spent.

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
  `TINE_MODEL_PRICES`. Streaming requests are never shadowed.
- Shadow results store no prompt or response text; the request record already holds the tenant's request
  and the frontier's response.

## Routing through a policy

With a route policy the proxy serves answers from a frozen policy. It is off unless configured, and a
tenant's provider has either a shadow policy or a route policy, never both. Run shadow mode first.

- `TINE_ROUTE_POLICIES`: path to a JSON file, tenant to provider to policy, as for shadow mode.
- `TINE_ANSWER_PATTERN`: how a comparable answer is read from a response, for agreement stages.
- `TINE_ROUTE_ESTIMATE_BASELINE`: `true` to estimate the frontier baseline for a routed answer (below).

For a non-streaming request that would have gone to the frontier, the proxy tries the policy's stages in
order, with the same request and the model swapped. A stage's models are called together. If the
stage's check passes, the first member's response is returned as it came from the provider, with
`x-tine-model` naming it and `x-tine-route: stage-N`. If no stage is accepted, or anything in the policy
path fails, the request goes to the frontier exactly as without a policy (`x-tine-route: frontier`).
Streaming requests and requests for an allowlisted non-frontier model are not routed.

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

The quality limit of a route policy is the one certified when it was frozen, on the tasks it was
frozen from. The proxy does not check that live traffic resembles those tasks. Outcomes still have to
be reported through the outcome endpoint for savings to count as verified.

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

- `webhook`: the proxy POSTs `{"tenant", "provider", "model", "request", "response"}` to a URL the
  operator controls and reads back `{"passed": true|false, "cost_usd": optional}`. This is how a
  tenant's own tests are run: beside their repository, on their infrastructure. **The proxy never
  executes model-written code.** `bearer_env` names an environment variable holding a token to send; the
  token is never written to the configuration.
- `judge`: the proxy asks a Claude model through its configured Anthropic provider whether the response
  resolves the request, with the instructions the measured judges used. `threshold` accepts on the
  judge's probability instead of its verdict. The judge's call is priced from `TINE_MODEL_PRICES`
  (`anthropic:<model>`) and counted as routing cost.

Every step fails closed: an error, a timeout, a malformed reply or a refusal rejects the answer, so the
request moves on. Limits:

- A configured check must do what the measured one did, or the policy's guarantee does not describe
  what runs. The measured judges read a GitHub issue and a unified diff
  (`results/2026-11-14-api-judge`); here the judge reads the request's user text and the response
  text. Its measured catch rate carries over only to traffic of that shape, and it was measured on
  public repositories the judge may have seen.
- A webhook receives the tenant's request and a model's response. Point it only at the tenant's own
  infrastructure.
- This path has been tested with stand-in providers only. It has not served real traffic.
