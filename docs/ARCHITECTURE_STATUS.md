# Architecture status

What exists, what has only been measured offline, and what is still a design. Dated
2026-11-17. Update it when a row changes. The target design is in `AGENTS.md`; this
page is the honest distance from it.

## In one paragraph

Tine today is an offline evaluation toolkit plus a development proxy. The toolkit
replays recorded benchmark answers to measure which routing ideas would save money
under a quality limit. The proxy keeps a ledger and sends every request to the
frontier model unless a frozen policy is configured; with one it can run the policy in
shadow mode or serve answers from it. That path is tested with stand-in providers and
a replay of recorded answers. It has also run with real Claude models on public
benchmark questions in the two live pilots (`results/2026-11-17-live-pilot`,
`results/2026-11-17-live-pilot-more`) and on a public set of one company's support
tickets (`results/2026-11-17-ticket-pilot`). It has never run on a customer's
traffic. No language model has been trained or fine-tuned (small classifiers were
trained on graded patches and did not work as a verifier), and no per-enterprise
server exists beyond the proxy.

## What is not being used

- **No MCP server.** Tine is not built on MCP and does not need it. Applications
  reach Tine by changing their API base URL to the proxy, which speaks the OpenAI and
  Anthropic HTTP APIs. (The Drive and Gmail connectors used during development are
  tools of the coding assistant, not part of Tine.)
- **Provider API keys in authorised paid runs only.** Most results come from answers
  already generated and graded in the LLMRouterBench and xRouteBench archives.
  Authorised paid runs called Claude models through the Anthropic API:
  - as a judge, $47.73 (`results/2026-11-14-api-judge`, `results/2026-11-16-judge-probes`,
    `results/2026-11-17-judge-prompt`);
  - to answer real requests through the proxy in the two live pilots, $23.65 and
    $60.00. That is $131.38 with the judge runs (`README.md`);
  - on 2026-11-17, $26.43 for the rate-limit tests and a 1,500-question shadow run
    (`results/2026-11-17-concurrency-and-long-shadow`) and $14.74 for the ticket pilot
    (`results/2026-11-17-ticket-pilot`). With these the results files add up to
    $172.55. Spend that is in no results file is not counted here.

  Open models of 14B and 32B were run locally and on Colab
  (`results/2026-11-11-judge-pilot`). The proxy has run with a real Anthropic key in
  those pilots, in-process and as `tine serve`. It has not run with a customer's key
  or traffic.
- **No model hosted or simulated as "the enterprise server".** There is no server
  holding company context, a request queue or a request history.
- **No language model inside the router.** See the next section.

## Is there an NLP or LLM in the routing model?

Not a language model. What exists:

| Piece | What it actually is |
| --- | --- |
| Prompt features | TF-IDF character n-grams, or fixed sentence embeddings computed once by an off-the-shelf encoder (MiniLM, Qwen3). The encoder is used as-is, never trained. |
| Predictors | Logistic regression, k-nearest-neighbours and a k-means cluster table, one per model, from scikit-learn. They predict "will this model get this request right". |
| Response check | A four-feature logistic regression per model on prompt and response token counts. |
| Agreement check | A comparison of two or three models' answers: exact text by default, or by value for LaTeX answers when the policy says `answer_compare: "math"` (`tine/policy.py`). |
| Judge | An LLM reading the issue and the patch. Measured with a Claude Code session (pilot), qwen2.5-coder 14B and 32B (neither works as a judge), and pinned Claude models through the API (`claude-sonnet-5-5` rejects 79% of wrong patches and 9% of right ones on 1,110 cases). The proxy can call it as a named check. |

So the only place an LLM appears is the judge. The proxy can call it as a named check,
but that path is tested with stand-ins only and has not served real traffic. A difficulty judge that scores the incoming request before any model runs,
and task splitting, are in the design and have not been built or tested.

## Layer 1: Gacrux, the routing model

| Component (design) | Status | Evidence so far |
| --- | --- | --- |
| Verified cache | Built in the proxy (exact match, per tenant, only after a confirmed success) | Tested with mocks. Savings not measured. |
| Non-AI tools (regex, SQL) | Not designed in the repo, not built | None |
| Difficulty judge on the request | Not built | None |
| Task splitting before routing | Not built | None |
| Fast classifier (predict from the prompt) | Built, offline only | Certifies about 0% savings on coding benchmarks at a 2% limit, with every feature set tried |
| Retrieval of similar past requests | Built as the kNN predictor, offline only | Same result. Benchmarks have no repeated requests, so this does not test a real company history |
| Small local judge | Built (`tine/judge.py`, `scripts/judge_patches.py`) and run | Open models of 14B and 32B do not work as a judge. A strong model does: `claude-sonnet-5-5` rejects 79% of the wrong patches that passed the free checks and 9% of the right ones on 1,110 cases (the 150-case pilot overstated this). See `results/2026-11-14-api-judge` |
| Try a cheaper model, check, escalate (chains) | Built offline; the proxy runs a certified chain as a frozen policy. Not in the original design | The approach that works: see `results/README.md` |
| Calibrated quality limit (epsilon, delta) | Built. The limit is certified offline and written into a frozen policy, which the proxy runs | Used by every result. Options for a relative rule and a net-loss guarantee were added 2026-11-11. In the live pilots policies certified at 4% from histories asked of real models; one 2% certificate passed on its last allowed loss and did not hold up (`results/2026-11-17-live-pilot`) |
| Trained or fine-tuned router weights | None | No language model has been trained. Small classifiers trained on graded patches did not work as a verifier (`results/2026-11-13-trained-verifier`) |

## Layer 2: the per-enterprise server

| Component (design) | Status |
| --- | --- |
| Drop-in proxy for OpenAI and Anthropic APIs, with streaming | Built for development. Sends everything to the configured frontier model unless a frozen policy is configured |
| Ledger of requests, costs and outcomes, with late outcome updates | Built on SQLite. Production Postgres not built |
| Tenant isolation | Built in the proxy, ledger and cache |
| Company context loaded once and cached | Not built. Its saving has not been measured |
| Request queue: batching, ordering, rate limits | Partly. No queue, batching or ordering, and no rate limit on requests. Built: a body-size limit, a per-tenant limit on shadow runs in flight, an optional per-minute cap on shadow calls, and a pause of shadow mode when a provider answers 429 or 529 (`tine/shadow.py`, `docs/PROXY.md`) |
| Per-company request history for retrieval | Not built (the ledger stores requests but nothing reads them for routing) |
| Shadow mode and the free savings report | Built (`tine/shadow.py`, `GET /v1/shadow`, and the report page in `docs/REPORT.md`). Tested with stand-in providers and a replay of recorded answers (`results/2026-11-16-shadow-replay`). Run with real Claude models on public benchmark questions in the two live pilots and on 1,500 more questions (`results/2026-11-17-concurrency-and-long-shadow`); never on a customer's traffic |
| Nightly retraining of the company's matrix | Not built |
| Live routing of a request to a cheaper model | Built for non-streaming requests through a frozen policy (`TINE_ROUTE_POLICIES`), with agreement stages and named checks (tests webhook, judge call). Agreement and single-model policies ran with real Claude models on public benchmark questions in the two live pilots; never on a customer's traffic. The named checks are tested with stand-in providers only |
| Random frontier holdout, to measure the saving under live routing (`TINE_AUDIT_RATE`) | Built. Run with real Claude models in the second live pilot (measured saving MedQA 56%, ARC 97%, FinQA 18%); never on a customer's traffic. It measures cost, not quality |
| Ledger audit: where each baseline came from, the policy digest and stage of each routed row, an append-only outcome history, an opt-in fail-closed ledger | Built and tested with stand-ins. No HTTP endpoint for the history |
| Check audit: what each step of a named check decided, per request | Built and tested with stand-ins. Read from the ledger row; not on the report page |
| Cost of streamed requests, read from the relayed events | Built and tested with stand-in streams; never run against a real provider's stream. Streams are not shadowed, routed or cached |
| Savings report page fields: freeze digest state, drift warning, shadow coverage, caller-supplied baselines, the holdout estimate | Built and tested with stand-in data (`docs/REPORT.md`). Never seen by a customer |
| Ledger backup while the proxy runs (`python -m tine backup-ledger`) | Built and tested; run in the image of the earlier Dockerfile beside a stand-in writer, not beside a real proxy |
| Live pilot harness (`scripts/live_pilot.py`) | Built, tested with a fake provider, and run with paid calls. A test tool, not part of the product |
| Docker deployment in the customer's cloud | Packaging written (`Dockerfile`, `docker-compose.yml`, `docs/DEPLOY.md`). An earlier Dockerfile was built and run once (40 paid requests, 2026-11-17); the current files have not been built |

## How the work is being done

1. **Data.** The LLMRouterBench archive: recorded answers, grades, costs and token
   counts from up to 38 models on 21 datasets. It lives on the development Mac and on
   Nandu's Google Drive, not in Git. The xRouteBench archive was added on 2026-11-14.
2. **Offline experiments.** `tine run`, `tine cascade` and the scripts in `scripts/`
   replay those records: fit a predictor or check on some tasks, certify a policy on
   others, score it on a third set. This runs on the Mac's CPU. No model is called.
3. **Colab** for the jobs the Mac cannot do: computing embeddings and hidden states
   on a GPU, running an open model as a judge, and running repository tests on
   recorded patches (which executes model-written code).
4. **Experiments that did use a model:** the judge check. First a pilot with
   sub-agents of a Claude Code session reading 150 patches, then open models, then an
   authorised paid run of pinned Claude models on 1,110 patches. And the live pilots
   of 2026-11-17, which asked real Claude models for a graded history and then sent
   new questions through the proxy.
5. **Results** go to `results/`, indexed in `results/README.md`.

## The gap between the evidence and the design

The benchmark evidence can test routing decisions on single, unrelated requests. It
cannot test the parts of the design that depend on one company's traffic: repeated
requests, shared context, per-company history, nightly retraining. Those need real
traffic. Shadow mode now exists to collect it. It has run with real models on public
benchmark questions and on a public set of one company's tickets, not on a customer's
traffic. The two early users are the way to get it.

## What would turn this into a running system

1. Done 2026-11-16: shadow mode in the proxy and a policy file the proxy can execute
   (`docs/PROXY.md`, `docs/POLICY.md`). Tested with stand-in providers, then run
   with real Claude models on public benchmark questions in the two live pilots
   (2026-11-17).
2. A first customer's real traffic in shadow mode.
3. A real outcome signal from the customer (tests passed, answer accepted).
4. More graded code tasks, so a chain with the judge certifies at a 2% limit.
5. Then context caching, the queue and nightly retraining on the per-enterprise server.
