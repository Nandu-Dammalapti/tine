# Architecture status

What exists, what has only been measured offline, and what is still a design. Dated
2026-11-11. Update it when a row changes. The target design is in `AGENTS.md`; this
page is the honest distance from it.

## In one paragraph

Tine today is an offline evaluation toolkit plus a development proxy. The toolkit
replays recorded benchmark answers to measure which routing ideas would save money
under a quality limit. The proxy forwards every request to the frontier model and
keeps a ledger. Nothing routes a live request to a cheaper model yet, no model has
been trained or fine-tuned, and no per-enterprise server exists.

## What is not being used

- **No MCP server.** Tine is not built on MCP and does not need it. Applications
  reach Tine by changing their API base URL to the proxy, which speaks the OpenAI and
  Anthropic HTTP APIs. (The Drive and Gmail connectors used during development are
  tools of the coding assistant, not part of Tine.)
- **No provider API keys in any experiment.** Every result in `results/` comes from
  answers that were already generated and graded in the LLMRouterBench archive. No
  model was called to produce them. The proxy accepts provider keys through
  environment variables, but it has only been run against mocked providers in tests.
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
| Judge | An LLM reading the issue and the patch. Piloted once with a Claude Code session as the judge. A script exists to run a local open model; it has not been run. |

So the only place an LLM appears is the judge, and it is not part of any running
system. A difficulty judge that scores the incoming request before any model runs,
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
| Small local judge | Script built (`scripts/judge_patches.py`), not run with a local model | A frontier-class judge rejected 95% of the wrong patches that passed the free checks, in a 150-case pilot |
| Try a cheaper model, check, escalate (chains) | Built, offline only. Not in the original design | The approach that works: see `results/README.md` |
| Calibrated quality limit (epsilon, delta) | Built, offline only | Used by every result. Options for a relative rule and a net-loss guarantee were added 2026-11-11 |
| Trained or fine-tuned router weights | None | No training has been run |

## Layer 2: the per-enterprise server

| Component (design) | Status |
| --- | --- |
| Drop-in proxy for OpenAI and Anthropic APIs, with streaming | Built for development. Sends everything to the configured frontier model |
| Ledger of requests, costs and outcomes, with late outcome updates | Built on SQLite. Production Postgres not built |
| Tenant isolation | Built in the proxy, ledger and cache |
| Company context loaded once and cached | Not built. Its saving has not been measured |
| Request queue: batching, ordering, rate limits | Not built |
| Per-company request history for retrieval | Not built (the ledger stores requests but nothing reads them for routing) |
| Shadow mode and the free savings report | Not built |
| Nightly retraining of the company's matrix | Not built |
| Live routing of a request to a cheaper model | Not built. `TINE_ALLOWED_MODELS` lets a client ask for another model; that is not routing |
| Docker deployment in the customer's cloud | Not built |

## How the work is being done

1. **Data.** The LLMRouterBench archive: recorded answers, grades, costs and token
   counts from up to 38 models on 21 datasets. It lives on the development Mac and on
   Nandu's Google Drive, not in Git.
2. **Offline experiments.** `tine run`, `tine cascade` and the scripts in `scripts/`
   replay those records: fit a predictor or check on some tasks, certify a policy on
   others, score it on a third set. This runs on the Mac's CPU. No model is called.
3. **Colab** for the two jobs the Mac cannot do: computing embeddings on a GPU, and
   running repository tests on recorded patches (which executes model-written code).
4. **One pilot that did use a model:** the judge check, with sub-agents of a Claude
   Code session reading 150 patches.
5. **Results** go to `results/`, indexed in `results/README.md`.

## The gap between the evidence and the design

The benchmark evidence can test routing decisions on single, unrelated requests. It
cannot test the parts of the design that depend on one company's traffic: repeated
requests, shared context, per-company history, nightly retraining. Those need real
traffic, which means shadow mode has to exist first. The two early users are the way
to get it.

## What would turn this into a running system

1. Shadow mode in the proxy: for a sample of requests, also call cheaper models and
   record outcomes, without changing what the user gets.
2. A policy file the proxy can execute: which models to try, which checks to apply,
   when to escalate. The offline tools already produce the pieces.
3. A real outcome signal from the customer (tests passed, answer accepted).
4. Then context caching, the queue and nightly retraining on the per-enterprise server.
