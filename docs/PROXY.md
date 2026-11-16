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
- `TINE_SHADOW_ANSWER_PATTERN`: optional regular expression; the comparable answer of a response is its
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
- **Agreement stages only.** A stage whose check needs the tenant's repository or a judge call cannot be
  shadowed by the proxy yet; such a policy is refused at startup.
- Projected savings use only requests whose costs are all known, so every stage model needs a price in
  `TINE_MODEL_PRICES`. Streaming requests are never shadowed.
- Shadow results store no prompt or response text; the request record already holds the tenant's request
  and the frontier's response.
