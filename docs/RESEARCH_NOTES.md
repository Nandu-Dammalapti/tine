# Research notes

Outside material the team has looked at, what it claims, how far to trust it, and
what it suggests for Tine. Vendor claims are recorded as claims. Nothing here is a
decision: proposals go to the team first. Add new notes at the top with the date read.

## Faros AI: token engineering and the Q3 2026 report (read 2026-11-10)

Faros sells an engineering-analytics platform that now covers AI token spend. It is
the closest public example of "cost per verified outcome" being sold to engineering
leaders, so it matters for positioning and for what an enterprise deployment is
expected to do.

### What Faros does

- **Measure.** "Token Intelligence" (announced 2026-06-17) attributes spend to
  sessions, PRs, teams and models, and labels tokens productive, inefficient
  (abandoned exploration loops) or wasteful (an expensive model on a trivial task).
  Headline metric: cost per verified outcome. The classification rules and the
  definition of "verified" are in a gated field guide, not public.
- **Route.** A "route" is model + harness + context + configuration. The "Time
  Machine" replays a customer's already-merged work through alternative routes and
  scores quality, cost and runtime. Changes are upgrade, downgrade or reconfigure,
  enforced through Faros's gateway or one the customer already runs.
- **Context.** Repository history, known bugs and standards are delivered to the
  agent before it starts (through MCP), to cut exploration loops.
- **Govern.** Budgets and quotas, allowed models and harnesses per team, policy
  violations, and an audit trail from shipped change to spend, model and context.

### Their routing evidence

Post dated 2026-06-25, updated 2026-07-24. 500 of Faros's own PR/Jira pairs from 12
repositories were curated; 211 remained after dropping any task where some route
lacked a valid result. Seven routes, one run each, scored by a blind LLM rubric judge.

| Route | Judge score | Cost per task | Cache share |
| --- | --- | --- | --- |
| Claude Code + GLM-5.2 | 0.568 | $0.92 | 89.7% |
| Claude Code + Kimi K2.6 | 0.566 | $1.78 | 90.0% |
| OpenCode + GLM-5.2 | 0.556 | $1.01 | 95.5% |
| Claude Code + Opus 4.8 | 0.521 | $1.76 | 99.7% |
| OpenCode + Kimi K2.6 | 0.514 | $0.86 | 94.8% |
| Codex + GPT-5.5 | 0.466 | $2.06 | 92.8% |
| OpenCode + Opus 4.8 | 0.454 | $1.05 | 100.0% |

Their policy: start with Claude Code + GLM-5.2; escalate to Claude Code + Kimi for
ambiguity, large diffs, high complexity or review risk. Their headline is "about 50%
lower cost per task at equal or better quality".

How far to trust it:

- The sample is curated, 289 of 500 tasks were dropped without stated reasons, and
  each task ran once per route.
- The judge is a model and was not validated against tests or the merged PR.
- Costs are projected from token counts and list prices, not bills.
- The 50% depends on the baseline: against OpenCode + Opus 4.8 it is 12%.
- The policy is static per kind of work. There is no per-request prediction and no
  bound on quality loss.
- It is Faros's own codebase and Faros sells the routing product.
- It disagrees with our data in one respect: there the open models match or beat the
  frontier routes on average; on LiveCodeBench gpt-5 is the most accurate model by
  6 points. Do not assume either generalises.

### The Q3 2026 report, "The Speed Trap"

Published 2026-09-18 as a takeaways post; the full report and its ten
recommendations are behind a download form and were **not read**. Telemetry from
22,000 developers on 4,000 teams with high AI adoption, most recent 12 months. It is
about delivery quality, not tokens.

| Finding | Number |
| --- | --- |
| Developers using an AI tool weekly | 79% |
| PRs merged without review | 31.3% in the prior report, now 76.3% |
| Work restarts per developer | up 66.7% |
| Time in QA | up 300.6% |
| Monthly incidents | up 125.4% |
| Average PR size | up 71.8% |
| PRs opened by agents at the leading edge | 13-14% |

The post says its findings are correlational and gives no method beyond the sample.

### What it suggests for Gacrux (proposals)

1. **Route as the unit, not model.** The same Opus 4.8 scored 0.521 in Claude Code
   and 0.454 in OpenCode. The decision matrix is models x task types; it would
   become routes x task types, with harness and reasoning effort recorded on every
   outcome row.
2. **Historical replay to bootstrap a tenant.** Replaying merged work gives every
   route's outcome on the same task without waiting for live traffic, which is the
   complete matrix calibration needs. It has to run where the customer's code and
   CI are, which is the case for the in-cloud server. Grade with the customer's
   tests, not a judge.
3. **Structured task features.** Faros escalates on complexity, diff size, work type
   and repository. Our prompt-only predictors cannot tell which tasks a cheaper
   model solves (see `results/README.md`). These fields are cheap to capture at the
   proxy and untested as predictor inputs.
4. **Cache-aware cost per task.** Cache share was 89.7-100% in their runs and one
   caching bug forced a rerun. This supports the existing AGENTS.md rules: keep a
   task or session on one model, include cache state in cost.
5. **"Merged" is a weak success label.** With 76.3% of PRs merged unreviewed, a
   merge proves little. The ledger already accepts late outcomes; the label should
   include restarts, reverts, QA failures and incidents.
6. **Waste categories in the savings report.** "Frontier used where a cheaper route
   was certified" is their "wasteful" category and is what shadow mode measures.
7. **Enterprise server.** Accept policies through a gateway the customer already
   runs; add per-team budgets and allowed-model lists to the audit ledger.

### Positioning

Faros recommends a static route per kind of work from a judge-scored replay, with no
guarantee. Tine's per-request decision with a certified bound on quality loss is the
part they do not offer, and the Q3 findings (quality slipping as speed rises) are an
argument for it. Do not claim Faros lacks a capability without checking their
current product.

### Sources

- <https://www.faros.ai/blog/ai-route-optimization> (2026-09-21)
- <https://www.faros.ai/blog/open-models-vs-frontier-models> (2026-06-25, updated 2026-07-24)
- <https://www.faros.ai/blog/token-intelligence-for-ai-engineering> (2026-06-17)
- <https://www.faros.ai/blog/ai-speed-trap-takeaways> (2026-09-18)
- <https://www.faros.ai/research/speed-trap>
- <https://www.faros.ai/platform>
- <https://www.faros.ai/solutions/optimal-model-routing>
- <https://www.faros.ai/blog/measuring-token-efficiency-ai-engineering> (2026-06-17)

## LLMRouter (ulab-uiuc), read 2026-11-14

Source: the repository README at <https://github.com/ulab-uiuc/LLMRouter> (MIT licence, PyPI
`llmrouter-lib`, paper arXiv:2608.06867). Read only; the library was not installed or run here, so
everything below is what the README states, not something we measured.

What it is: an open-source collection of about 19 routers behind one command line (`llmrouter
train / infer / chat / serve`), a data pipeline that runs candidate models on 11 benchmarks, a
benchmark of recorded outcomes (xRouteBench: 8 datasets, 18 models), and an OpenAI-compatible server.

| Part | Use for Tine | Why |
| --- | --- | --- |
| Single-round routers: kNN, SVM, MLP, matrix factorisation, Elo, RouterDC, GraphRouter, Hybrid LLM, CausalLM | **Baselines only** | All choose a model from the prompt alone. Our result is that prompt-only routing certifies 0-5% savings on coding with every representation tried. Worth running through our certifier as we did for Avengers-Pro, to say so against a public library. |
| AutoMix | **Worth testing** | The one cascade in the set: a small model answers, checks its own answer, and escalates. It is the published version of our "try, check, escalate". We found small models cannot check code; test it on exact-answer tasks. |
| xRouteBench | **Worth pulling** | Recorded responses, scores and token counts per model per query. Responses are what agreement checks need; more datasets to test agreement on. Check licence and overlap with LLMRouterBench first. |
| `llmrouter serve` (OpenAI-compatible, streaming, `/v1/responses`) and the plugin folder for custom routers | **Reference, and a possible distribution route** | Shows the proxy surface customers expect. Gacrux could be registered as a custom router for side-by-side comparison. Not our runtime: no ledger, tenant isolation, or guarantee. |
| Router-R1, multi-round and agentic routers | **Not for us now** | An LLM does the routing over several rounds, on a GPU, with pinned vLLM. Routing cost is high and it is netted from savings (AGENTS.md section 2). |
| GMTRouter, PersonalizedRouter | **Not for us** | Per-user preference. Tine learns per company from verified outcomes. |
| TSRouter, ComfyUI nodes, Slack and Discord gateway | **Not for us** | Multimodal and chat tooling outside the product. |

What it does not have, which is Tine's ground: no quality limit and no calibration (routers are
trained on `alpha * performance - beta * cost`), no check on the answer after it is produced other
than AutoMix's self-check, no savings ledger, no learning from a company's own outcomes. Its headline
(learned routers beat the best fixed model by 14.6% relative) is a performance figure on QA-style
benchmarks with open models, not cost saved at held quality, and it has no SWE-bench.

Reading for us: the "which router" layer is a commodity that anyone can install. That supports
putting Tine's effort into verification, the guarantee and per-company outcomes, and using this
library as the baseline set rather than rebuilding those routers.

Follow-up, 2026-11-14: the agreement check was run on xRouteBench and five of the library's routers
were run on our LiveCodeBench pool. Results: `results/2026-11-14-xroutebench`.
