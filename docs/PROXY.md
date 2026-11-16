# Local proxy and ledger

The development proxy exposes native OpenAI `POST /v1/chat/completions` and Anthropic `POST /v1/messages` routes, including streaming passthrough. It uses an authenticated tenant mapping and sends every request to that provider's configured frontier model unless a frozen policy is configured. With one, it can run the policy in shadow mode (report only) or serve answers from it; see Shadow mode, Routing through a policy and Named checks below, which also list `TINE_SHADOW_POLICIES`, `TINE_SHADOW_SAMPLE_RATE`, `TINE_SHADOW_MAX_COST_USD`, `TINE_SHADOW_MAX_CONCURRENT`, `TINE_SHADOW_MAX_UNKNOWN_COST_CALLS`, `TINE_ROUTE_POLICIES`, `TINE_ROUTE_ESTIMATE_BASELINE`, `TINE_ROUTE_DEADLINE_SECONDS`, `TINE_CHECKS`, `TINE_TENANT_CHECKS`, `TINE_ANSWER_PATTERN` and `TINE_ALLOWED_MODELS`. Nothing is enabled by an offline experiment alone.

To run the proxy in a container, see [DEPLOY.md](DEPLOY.md). The image has not been built or run yet.

Use the activated Python environment from the README. Configure these variables through your shell's secure configuration or environment settings:

| Variable | Meaning |
| --- | --- |
| `TINE_TENANT_KEYS` | JSON object mapping incoming proxy tokens to tenant IDs |
| `TINE_OPENAI_API_KEY` / `TINE_ANTHROPIC_API_KEY` | Provider credentials; configure only the providers you use |
| `TINE_OPENAI_FRONTIER_MODEL` / `TINE_ANTHROPIC_FRONTIER_MODEL` | Exact frontier model name for each provider |
| `TINE_LEDGER_PATH` | SQLite file, default `artifacts/ledger.sqlite3` |
| `TINE_OPENAI_BASE_URL` / `TINE_ANTHROPIC_BASE_URL` | Optional API-compatible upstreams; defaults are official HTTPS endpoints |
| `TINE_TIMEOUT_SECONDS` | Upstream timeout, default 30 |
| `TINE_MODEL_PRICES` | Explicit USD-per-million token rates keyed by `provider:model`. Optional, except that shadow sampling needs a price for every stage model and judge model of a shadow policy |
| `TINE_VERIFIED_CACHE` | `true` by default; cache requires an explicit successful outcome |

Never commit credentials or paste provider keys into notebook cells. The app refuses startup without configured tenant keys. Clients use their **proxy tenant token**, not the provider key, in `Authorization: Bearer ...`; Anthropic clients can use `x-api-key` for that proxy token. Outbound requests use the server's provider credential. A caller-supplied tenant header cannot select a tenant.

Start locally:

```bash
python -m tine serve --host 127.0.0.1 --port 8000
```

`GET /health` reports the routing mode (`frontier-default`, or `policy` when a route policy is configured), whether shadow mode is sampling, and SQLite development storage. Provider routes return a Tine request ID in `x-tine-request-id`. After an objective check completes, authenticate as the same tenant and update it with `PATCH /v1/outcomes/{request_id}` and JSON `{"success": true}` or `{"success": false}`. Optional `cost_usd`, `routing_cost_usd`, and `baseline_cost_usd` must come from actual accounting; they are not guessed. `GET /v1/savings` returns that tenant's ledger summary.

Record the returned answer's generation cost in `cost_usd` and the additional routing cost in `routing_cost_usd`. Routing cost includes earlier attempts that escalated, judges, tests, embeddings, and allocated customer-hosted compute. Include each expense once. If an older caller included these expenses in `cost_usd`, split that amount before supplying the routing component. `baseline_cost_usd` is the comparable always-frontier cost.

For example, a verified answer costing $0.25 plus $0.90 in routing against a $1.00 baseline has **-$0.15 net savings**. Send `{"success": true, "cost_usd": 0.25, "routing_cost_usd": 0.90, "baseline_cost_usd": 1.00}`. Omitting a cost on later updates preserves it; explicit `null` marks it unknown again. Zero must be an explicit accounting result.

Without a route policy, automatic proxy records leave routing cost unknown, including cache hits: no provider call does not imply free checking or infrastructure. With a route policy the proxy records routing cost itself for requests the policy handled (see Routing through a policy); cache hits still leave it unknown. Cache hits do not copy the source request's routing cost. Older SQLite ledgers gain a nullable routing-cost column; existing rows stay unknown until costs are supplied. Restart all writers together when upgrading. Cache eligibility still depends on the verified outcome, independently of whether costs have arrived.

Net savings require known answer, routing, and baseline costs for a verified successful request. `verified_cost_usd` now includes both cost components, exposed separately as `verified_answer_cost_usd` and `verified_routing_cost_usd`. `verified_savings_usd` and `savings_fraction` are null when no comparable request exists. `unknown_cost_requests` now counts requests missing either component; separate answer/routing counts identify the gaps. Known cost totals sum measured components across successes, failures, and pending requests, so they are incomplete while any component is unknown. Cost per verified success is null until every request has both costs. Unrepresentable numeric aggregates are also null.

Unknown outcomes and costs remain unknown. Cached responses are scoped by tenant, provider, upstream endpoint, model, exact request contents, and forwarded Anthropic version/beta headers. With a route policy the cached answer may be a cheaper model's: a routed answer is stored under the request as it arrived (see Routing through a policy). They become eligible after success is explicitly confirmed. After upgrading, responses recorded with the older cache keys require a fresh upstream request and successful outcome before cache reuse, because those keys lack upstream context. Streaming responses are not cached. The ledger stores prompts/responses; use only data you intend to retain in the configured local database.

For OpenAI-compatible usage, omitted optional cache details retain the zero-cache default. Explicitly null or malformed `prompt_tokens_details` instead leave cached tokens, estimated cost, and the automatic frontier baseline unknown, even if input/output totals are valid. The provider response still passes through unchanged; a later measured cost can be supplied through the outcome endpoint.

Late outcome reversals or newer resolved failures invalidate prior cached successes. Resolved outcomes use a transactionally assigned database sequence, so equal or backward-moving wall-clock timestamps cannot revive stale success evidence. Existing ledgers are upgraded automatically; legacy entries retain their prior timestamp order until new sequenced evidence arrives. Restart all proxy writers together when upgrading so every writer uses this ordering. If local ledger recording fails, a successful provider response is still returned with `x-tine-ledger: unavailable`. Net savings reports cover comparable, verified successful requests only; they are not billing amounts or proof of end-to-end savings. Failed and pending requests with known costs contribute to total spend, including routing costs, and to the cost-per-verified-success metric.

`TINE_ALLOWED_MODELS` can explicitly allow client-requested development alternatives, with a single frontier retry on an upstream failure before a response is emitted. Supply a JSON object of provider names and model-name arrays, for example `{"openai": ["EXACT_MODEL_NAME"]}`. Names match exactly. A string in place of an array, an unknown provider key, or an empty model name causes startup to fail before a ledger is opened. This switch is not a calibrated router. Failure after streaming starts cannot safely be retried without duplicating output. Keep it unset for frontier-only operation.

This is a development implementation: SQLite is not the planned production Postgres backend. Tests use mock upstreams, not paid API calls. Do not deploy it as a production guarantee or host it on Colab. Container packaging and its limits are in [DEPLOY.md](DEPLOY.md). No live provider credentials were requested or validated during this work.

## Shadow mode

Shadow mode shows what a frozen policy ([POLICY.md](POLICY.md)) would have saved on a tenant's real
traffic, without changing anything the tenant sees. It is off unless configured.

- `TINE_SHADOW_POLICIES`: path to a JSON file mapping tenant to provider to policy, for example
  `{"acme": {"openai": { ...policy... }}}`. The policy's `frontier` must be that provider's configured
  frontier model, and its `tenant` must be the tenant it is listed under: a policy frozen for one
  tenant cannot be given to another, and a policy that names no tenant is refused.
- `TINE_SHADOW_SAMPLE_RATE`: the share of eligible requests to shadow, 0 to 1.
- `TINE_SHADOW_MAX_COST_USD`: required when sampling. Shadow runs stop for a tenant once its known
  shadow spend reaches this amount. See "The spending cap" below for what it does and does not bound.
- `TINE_SHADOW_MAX_CONCURRENT`: shadow runs in flight at once, for the whole proxy. Default 8. A sampled
  request that arrives while all are busy is not shadowed.
- `TINE_SHADOW_MAX_UNKNOWN_COST_CALLS`: how many shadow calls or checks of unknown cost a tenant may
  have before its shadowing stops. Default 20.
- `TINE_ANSWER_PATTERN`: optional regular expression; the comparable answer of a response is its
  last match (first group if there is one). Without it, whole response texts are compared. A policy
  may carry its own `"answer_pattern"` key, which is used for that policy in place of this one.

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

The same report as a page to send to a customer:
`python scripts/savings_report.py --ledger LEDGER --tenant TENANT --policy POLICY.json --output report.html`
([REPORT.md](REPORT.md)). `--note` adds one line under the headline.

### The spending cap

- The cap counts known costs. At startup the proxy refuses to sample unless every stage model of every
  shadow policy has an input and output price in `TINE_MODEL_PRICES`, and every judge step those
  policies use has an `anthropic:<model>` price.
- A call can still have an unknown cost: it failed or timed out, the reply carried no usage, or a
  webhook reported a cost that is not a usable amount. Such calls cannot be counted in dollars. They
  are counted by number, and shadowing stops for the tenant once there are more than
  `TINE_SHADOW_MAX_UNKNOWN_COST_CALLS`. What those calls cost is not bounded in dollars by anything here.
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
  `TINE_MODEL_PRICES`. Streaming requests are never shadowed.
- Shadow results store no prompt or response text; the request record already holds the tenant's request
  and the frontier's response.

## Routing through a policy

With a route policy the proxy serves answers from a frozen policy. It is off unless configured, and a
tenant's provider has either a shadow policy or a route policy, never both. Run shadow mode first.

- `TINE_ROUTE_POLICIES`: path to a JSON file, tenant to provider to policy, as for shadow mode.
- `TINE_ANSWER_PATTERN`: how a comparable answer is read from a response, for agreement stages.
- `TINE_ROUTE_ESTIMATE_BASELINE`: `true` to estimate the frontier baseline for a routed answer (below).
- `TINE_ROUTE_DEADLINE_SECONDS`: how long the cheaper stages may take together for one request. Default:
  `TINE_TIMEOUT_SECONDS`. When it runs out the request goes to the frontier.

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
  baseline supplied later through the outcome endpoint replaces the estimate and counts as measured.
- If the frontier cannot be reached after the policy escalated, the 502 row still records what the
  cheaper attempts and checks cost.

The quality limit of a route policy is the one certified when it was frozen, on the tasks it was
frozen from. The proxy does not check that live traffic resembles those tasks. Outcomes still have to
be reported through the outcome endpoint for savings to count as verified.

Limits of routing and shadow mode that are not fixed in code:

- **Answers are not read the way they were when the policy was certified.** The multiple-choice policy
  was certified on answers read by `scripts/import_xroutebench.py`: the last letter in brackets, upper
  case; or the last number, with commas and trailing zeros removed; or the last `\boxed{}`. The proxy
  reads an answer with one regular expression and compares the raw match. The policy file does not say
  which reader it was certified with. Two effects follow. With a pattern narrower than the certified
  reader (one that accepts `(A)` but skips `(b)`), the proxy can accept answers the certified check
  would reject. With the closest pattern, answers that differ only in form (`(b)` and `(B)`, `1,000.50`
  and `1000.5`) do not agree, the request pays every cheap call and then the frontier, and live savings
  fall below the stored estimate. Compare the acceptance share in shadow mode with the policy's
  estimate before routing. The replay in `results/2026-11-16-shadow-replay` does not test this: its
  stand-in provider returns answers that were already read.
- **A request with tools still goes through agreement stages.** Replies with tool calls never agree, so
  such a request pays the cheap calls and then the frontier.
- **No memory of failures between requests.** While a cheap model or check is down, every routed
  request still tries it and waits up to the deadline. There is no circuit breaker. The members of a
  stage are awaited together, so one slow member holds the stage even when another has already failed.
- **Nothing ties a policy file's stages to its certification.** The proxy checks that the file says
  `certified`, that its bound is not above its epsilon when both are given, that it names the tenant
  it is configured for, and that no stage repeats a model. It does not detect a file whose stages,
  checks or models were edited after freezing. `task_type` is not read: the proxy cannot tell what
  kind of task a request is, so every routed request of a tenant and provider goes through that
  tenant's one policy for the provider.

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
  token is never written to the configuration. `timeout_seconds` defaults to 60 and may be at most 600.
  A `cost_usd` that is not a usable amount (negative, `NaN`, infinite, or above one million dollars)
  makes that check's cost unknown. It is never added to a total.
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
  longer text is not cut: the judge is not asked, the answer is rejected and the request moves on. Long
  conversations therefore never pass a judge step. System, assistant and tool messages are not shown to
  the judge.
- The text the judge reads is not trusted. The tags that frame it (`<issue>`, `<patch>` and their
  closing forms) are made inert wherever they occur inside it, so text cannot close its frame and
  continue as if it were instructions.
- A reply from the judge that holds verdict objects that disagree is not a verdict, and rejects.
- The judge is told that the tests passed only when a webhook step passed just before it in the same
  check. In a check with no webhook step it is told the patch has not been applied or tested. That
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
  instructions to ignore. Whether text that argues with the judge can sway it has not been measured: no
  injected cases were run. Measure them before relying on a judge step for live routing.
- This path has been tested with stand-in providers only. It has not served real traffic.
