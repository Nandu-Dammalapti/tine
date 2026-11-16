# Architecture status

What exists, what has only been measured offline, and what is still a design. Dated
2026-11-16. Update it when a row changes. The target design is in `AGENTS.md`; this
page is the honest distance from it.

## In one paragraph

Tine today is an offline evaluation toolkit plus a development proxy. The toolkit
replays recorded benchmark answers to measure which routing ideas would save money
under a quality limit. The proxy keeps a ledger and sends every request to the
frontier model unless a frozen policy is configured; with one it can run the policy in
shadow mode or serve answers from it. That path is tested with stand-in providers and
a replay of recorded answers only: nothing has served real traffic. No language model
has been trained or fine-tuned (small classifiers were trained on graded patches and
did not work as a verifier), and no per-enterprise server exists beyond the proxy.

## What is not being used

- **No MCP server.** Tine is not built on MCP and does not need it. Applications
  reach Tine by changing their API base URL to the proxy, which speaks the OpenAI and
  Anthropic HTTP APIs. (The Drive and Gmail connectors used during development are
  tools of the coding assistant, not part of Tine.)
- **Provider API keys in one experiment only.** Results come from answers already
  generated and graded in the LLMRouterBench and xRouteBench archives. The exception
  is the judge: an authorised paid run called Claude models through the Anthropic API
  ($12.69, `results/2026-11-14-api-judge`), and open models of 14B and 32B were run
  locally and on Colab (`results/2026-11-11-judge-pilot`). The proxy accepts provider
  keys through environment variables, but it has only been run against mocked
  providers.
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
| Agreement check | A string comparison of two or three models' answers. |
| Judge | An LLM reading the issue and the patch. Measured with a Claude Code session (pilot), qwen2.5-coder 14B and 32B (neither works as a judge), and pinned Claude models through the API (`claude-sonnet-5-5` rejects 79% of wrong patches and 9% of right ones on 1,110 cases). The proxy can call it as a named check. |

So the only place an LLM appears is the judge. The proxy can call it as a named check,
but that path has not served real traffic. A difficulty judge that scores the incoming request before any model runs,
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
| Try a cheaper model, check, escalate (chains) | Built, offline only. Not in the original design | The approach that works: see `results/README.md` |
| Calibrated quality limit (epsilon, delta) | Built, offline only | Used by every result. Options for a relative rule and a net-loss guarantee were added 2026-11-11 |
| Trained or fine-tuned router weights | None | No language model has been trained. Small classifiers trained on graded patches did not work as a verifier (`results/2026-11-13-trained-verifier`) |

## Layer 2: the per-enterprise server

| Component (design) | Status |
| --- | --- |
| Drop-in proxy for OpenAI and Anthropic APIs, with streaming | Built for development. Sends everything to the configured frontier model unless a frozen policy is configured |
| Ledger of requests, costs and outcomes, with late outcome updates | Built on SQLite. Production Postgres not built |
| Tenant isolation | Built in the proxy, ledger and cache |
| Company context loaded once and cached | Not built. Its saving has not been measured |
| Request queue: batching, ordering, rate limits | Not built |
| Per-company request history for retrieval | Not built (the ledger stores requests but nothing reads them for routing) |
| Shadow mode and the free savings report | Built (`tine/shadow.py`, `GET /v1/shadow`, and the report page in `docs/REPORT.md`). Tested with stand-in providers and a replay of recorded answers (`results/2026-11-16-shadow-replay`). Never run on real traffic |
| Nightly retraining of the company's matrix | Not built |
| Live routing of a request to a cheaper model | Built for non-streaming requests through a frozen policy (`TINE_ROUTE_POLICIES`), with agreement stages and named checks (tests webhook, judge call). Tested with stand-in providers only |
| Docker deployment in the customer's cloud | Packaging written (`Dockerfile`, `docker-compose.yml`, `docs/DEPLOY.md`). The image has never been built or run |

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
   authorised paid run of pinned Claude models on 1,110 patches.
5. **Results** go to `results/`, indexed in `results/README.md`.

## The gap between the evidence and the design

The benchmark evidence can test routing decisions on single, unrelated requests. It
cannot test the parts of the design that depend on one company's traffic: repeated
requests, shared context, per-company history, nightly retraining. Those need real
traffic. Shadow mode now exists to collect it, but it has not been run on any. The
two early users are the way to get it.

## What would turn this into a running system

1. Done 2026-11-16: shadow mode in the proxy and a policy file the proxy can execute
   (`docs/PROXY.md`, `docs/POLICY.md`). Tested with stand-in providers only.
2. A first customer's real traffic in shadow mode.
3. A real outcome signal from the customer (tests passed, answer accepted).
4. More graded code tasks, so a chain with the judge certifies at a 2% limit.
5. Then context caching, the queue and nightly retraining on the per-enterprise server.
