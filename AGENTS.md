# Tine: project context for coding agents

Read this file before writing code. It records the project direction supplied by the team in November 2026. Product goals and research references describe intended behavior, not capabilities already implemented or independently verified claims.

## Current scope

This repository holds development prerequisites, the Gacrux v1 evaluation path on CPU (`tine/`, tests in `tests/`), and Colab notebooks for GPU work (`colab/`). NEXT_STEPS.md records status and the next tasks. CPU training of small routers (kNN, logistic regression, boosted trees) on local data is in scope. Model downloads, GPU jobs, inference servers, bulk dataset downloads, and paid API calls still need a task requesting them. Do not expand into the proxy or other product pieces without a task requesting it.

Use the existing checkout. Cloud tasks are already isolated; do not create a Git worktree unless the user explicitly requests one. Inspect existing files and preserve user changes before modifying them. Never commit virtual environments, caches, credentials, customer prompts, model weights, or generated benchmark outputs.

## 1. What Tine is

Tine routes AI requests to the cheapest model likely to get them right, measures whether quality held up, and learns from each company's own work.

The problem is increasing AI spend, especially from agents. Teams often use a frontier model for everything because they cannot establish that cheaper models will succeed on their tasks.

The intended product includes:

- A drop-in proxy compatible with OpenAI and Anthropic APIs, including streaming. A customer should be able to change the base URL without breaking existing integrations.
- A logger for requests, responses, selected models, actual costs, and outcomes such as tests passing or an answer being accepted.
- Shadow mode that samples real traffic, privately compares cheaper models, and produces a free savings report before paid routing.
- The Gacrux router, which chooses models subject to a calibrated quality constraint.
- A verified answer cache for requests whose matching answers have already passed their checks.
- A savings ledger recording routing decisions, costs, and outcomes for finance and audit.
- Later, distillation of repeated verified tasks into small customer-owned models.

Deployment is a Docker container in the customer's cloud. Easy work can go to open models on customer hardware; difficult work can go to frontier APIs. No third-party router should receive prompts. Prompts sent to a frontier provider still reach that provider: never claim that all data stays in-house.

The proposed business model is 20% of verified savings against the customer's default frontier model. Shadow mode and the report are free. At $1 million annual spend and 40% savings, that would be $80,000 annual revenue for Tine.

The team is Nandan (Nandu) Dammalapati, Ashmith Yaramada, and Nirvika Choudhury. All three write code; the team is applying to YC.

## 2. Core objective

This is constrained cost minimization. For request `x`, choose the cheapest model `m` satisfying `P(success | x, m) >= tau`. If no model qualifies, use the frontier fallback.

Choose `tau` so quality loss versus always using the frontier model is at most `epsilon`, with confidence `1 - delta`.

| Parameter | Meaning | Source |
| --- | --- | --- |
| `epsilon` | Maximum quality loss the customer accepts | Business choice agreed with the customer |
| `delta` | Probability that the guarantee fails | Business choice |
| `tau` | Required confidence before using a cheaper model | Computed from held-out calibration data |

Lower `tau` routes more traffic to cheaper models, increasing savings and risk. Higher `tau` is more cautious. Do not increase `epsilon` to inflate savings. Improve the predictor so the calibrated threshold can decrease at the same quality constraint.

Initial planning values are `epsilon = 0.01–0.02` and `delta = 0.05` for coding, and about `epsilon = 0.05` for low-stakes internal work. These are proposed settings, not an established guarantee.

Measure cost per completed task, including retries, output length, and caching. Per-token prices alone are insufficient.

## 3. Gacrux architecture

Process each request through these stages:

1. **Verified cache.** Return an exact or appropriately validated near-exact cached answer only when its required check already passed. Similarity alone does not establish correctness.
2. **Fast classifier.** Use prompt embeddings and cheap features to predict success probability for each candidate model. This is a CPU path targeting under 10 ms and roughly 90% of traffic.
3. **Retrieval memory.** Retrieve about 20 similar requests for the same tenant. Use which model handled each request and whether it succeeded. Never learn only from previous routing choices, which would repeat past mistakes.
4. **Small local judge.** When the previous stages remain uncertain, use a customer-hosted open model through vLLM. Use prefix caching and continuous batching; this is intended for only about 5–10% of traffic.
5. **Route and record.** Call the chosen model, log cost and outcome, and feed verified outcomes back into future training data.

The decision matrix is a models-by-task-types table of predicted success rates and costs. The classifier and retrieval memory inform it for new requests. Nightly updates retrain each tenant's classifier/matrix and recalibrate `tau` on held-out data.

Hard constraints:

- Cache, classifier, and retrieval stages together should add under 20 ms. The local judge is the slow path.
- Customer data, retrieval memory, models, and caches must remain isolated by tenant. Do not expose one team's prompts to another team's requests.
- Shared company context means static company information, stack, and conventions cached as a prefix. It is not a shared running log of prompts.
- Include cache state in cost estimates. Cached input can cost about 10% of normal input, depending on provider; verify current pricing. Switching models mid-session can discard that benefit.
- Agent calls can be routed individually when appropriate, but keep a task or session on one model when cache economics dominate. Keep each subtask on one model to preserve caching.
- Explicit strong-model planning followed by cheap-model execution is a later feature.
- On router errors or timeouts, attempt the frontier fallback. Tine's routing layer must not introduce an avoidable request failure; provider failures remain possible.

## 4. Calibration and the quality guarantee

Reserve a calibration set never used for training: initially at least a few hundred examples per task type, with sample size justified against the desired bound.

Calibrate raw classifier scores first, using temperature scaling or isotonic regression. For candidate thresholds from low to high, route the calibration set and measure quality loss against the always-frontier baseline. Use a valid upper confidence bound, such as Clopper–Pearson for a suitable binomial quantity or Hoeffding for a suitably bounded loss, rather than only the empirical average.

Choose the lowest threshold whose upper bound is at most `epsilon`. If none qualifies, force all traffic to the frontier fallback. The conceptual `tau = 1.0` sentinel must not accidentally permit a cheaper model with score exactly 1.0.

Testing many thresholds requires a justified correction, such as Learn-then-Test or conformal risk control. A naive loop with an uncorrected bound at each threshold does not establish the claimed global confidence. Define the loss, sampling assumptions, and applicability of the bound before claiming a guarantee.

Suggested reading supplied by the team: *Conformal LLM Routing with Distribution-Free Safety Guarantees* (Uddin & Bauer, ACL 2026 SRW), described as using a logistic-regression embedding gate with Clopper–Pearson calibration. Verify the paper and exact assumptions before relying on it.

## 5. Data and outcomes

Use existing graded data first; do not pay to regenerate available results.

- [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) is the proposed main bootstrap source, described by the team as 400K+ instances across 21 datasets and 33 models. Verify current contents, licenses, and schemas before use.
- RouteLLM Chatbot Arena preferences and RouterBench are additional candidate sources.
- Coding tasks with executable checks, including HumanEval, MBPP, and SWE-bench Lite, provide objective labels and are the first target domain.

Only fill genuine gaps later: batch open models such as Qwen, Llama, or Gemma through vLLM; call frontier models through official APIs. Grade coding with tests and other work with a justified judge.

The proposed result row contains:

```text
tenant, task_id, prompt, task_type, model, success, cost_usd,
input_tokens, output_tokens, cached_tokens, latency_ms, timestamp
```

Production signals include tests passing, CI succeeding, an answer being accepted, no regeneration, and a ticket staying closed. Some arrive days later, so outcomes must be updateable. Distinguish unknown/pending outcomes from failure and success.

## 6. Baselines and evaluation

Always compare against:

- **Best single model:** always use the frontier baseline.
- **Random router.**
- **kNN router:** a strong simple baseline.
- **Avengers-Pro:** embedding, clustering, and per-cluster accuracy/cost scoring; the main proposed routing baseline to reproduce.
- **Oracle:** the cheapest model that succeeded, chosen in hindsight. This is an unattainable runtime ceiling, not a deployable router.

Report percent cost saved subject to at most `epsilon` quality loss, cost-quality curves against all baselines, the fraction of the oracle gap closed, router p50/p99 latency, and whether held-out quality loss respects the intended confidence statement.

Team-supplied research pointers include *When Simple kNN Beats Complex Learned Routers* ([arXiv:2505.12601](https://arxiv.org/abs/2505.12601)) and LLMRouterBench. The team reports that many routers fail to beat the best single model, that the best benchmark result saved up to 31.7% at equal accuracy, and that outcome data matters more than embedding choice. Verify these claims, dates, and experimental settings before citing them publicly.

## 7. Build order and compute

The proposed demo path is steps **1 → 2 → 3 → 4 → 6 → 8 → 10**, filling missing answers only where necessary. Steps 7 and 9 are deferred.

| Step | Work | Intended runtime |
| --- | --- | --- |
| 1 | Load existing router benchmarks and coding tasks | CPU |
| 2 | Generate missing open-model answers with vLLM | Colab A100 / G4 |
| 3 | Generate missing frontier answers through APIs | CPU + API credits |
| 4 | Grade answers with tests or a judge | CPU; API/local judge where needed |
| 5 | Embed prompts once and persist results | Colab T4 / L4 |
| 6 | kNN + logistic regression/boosted trees; reproduce Avengers-Pro | CPU |
| 7 | Fine-tuned encoder with per-model heads, later | Colab L4, or TPU with JAX |
| 8 | Calibrate and compute the safe threshold | CPU |
| 9 | Distillation/LoRA on permitted open-model outputs, later | A100 / G4 / suitable JAX TPU |
| 10 | Cost-quality charts against every baseline | CPU |

Step 6 needs existing or newly produced embeddings; the demo's omitted embedding step is not permission to assume embeddings are available. Inspect dataset artifacts first.

After the evaluation path, implement the FastAPI proxy, logger, shadow mode, and savings report on a small VM. Never host the production proxy on Colab; Colab is for batch work.

The November 2026 team plan is to share initial Gacrux decision-matrix iterations from GitHub while Nandu gathers data, benchmarks routers, and reruns experiments on Colab. Treat it as planning context, not completed work.

Colab rules: debug with about 50 rows on CPU before requesting a GPU; checkpoint to Google Drive; disconnect and delete the runtime immediately after a job; use vLLM batching rather than per-example generation loops; use TPUs with JAX. Compute and paid API use require the task's authorization.

## 8. Suggested stack and layout

The proposed stack is Python/FastAPI for the proxy; Postgres for ledger/outcome storage; tenant-partitioned pgvector or FAISS for retrieval; scikit-learn/LightGBM for v1; sentence-transformers for embeddings; PyTorch or JAX for v2; vLLM for local inference; Docker for customer deployment; and a simple savings-report page.

Suggested future packages are `proxy/`, `router/`, `calibration/`, `data/`, `eval/`, `shadow/`, `ledger/`, and `notebooks/`. Keep Colab notebooks thin and import shared implementation from packages. These directories are a plan, not a requirement to scaffold unused modules now.

## 9. Provider terms and data policy

The team flags restrictions in Anthropic and OpenAI terms on using outputs to train competing models. Have the team review current provider terms before distillation or bulk generation. Distillation is intended to use permitted open-weight outputs; closed-model outputs are intended only for grading and comparison, subject to applicable terms.

Using provider outputs as success labels for a routing classifier may have a different policy treatment from training an LLM; do not assume permission without checking current terms. Use official APIs for bulk generation, never scripted consumer chat applications.

## 10. Prior art and positioning

Team-supplied prior-art pointers, which should be checked before external claims:

| Idea | Examples |
| --- | --- |
| Small LLM as a router | Katanemo Arch-Router |
| kNN routing | arXiv:2505.12601 |
| Company-specific routing | Not Diamond custom routers |
| Conformal quality guarantees | Uddin & Bauer, ACL 2026 |
| Cost-quality routing | FrugalGPT, RouteLLM, Avengers-Pro, OpenRouter auto |
| Distillation from production traffic | OpenPipe |

Tine's proposed distinction is the combination: automatic learning from production outcomes, deployment in the customer's cloud, calibrated quality constraints, nightly per-company retraining, and later distillation. Do not claim uniqueness or competitor capabilities without current evidence.

## 11. Glossary

- **Gacrux:** Tine's routing and decision-matrix system.
- **Shadow mode:** sampled parallel model comparisons hidden from the end user.
- **Oracle:** hindsight selection of the cheapest model that succeeded.
- **Best single:** the always-frontier comparison baseline.
- **Tenant:** one customer company, with isolated data and routing state.
