# Tine: project context for coding agents

Read this file before writing code. It records the project direction supplied by the team in November 2026. Product goals and research references describe intended behavior, not capabilities already implemented or independently verified claims.

## Current scope

This repository began as development prerequisites and smoke checks. Later requests widened it to the offline toolkit, the development proxy, shadow mode and frozen policies described in section 12. Model downloads, Colab compute, bulk dataset downloads and paid API calls still need the task's explicit authorisation, and paid calls need a spending cap. Do not expand scope without a task requesting it.

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

The proposed business model is 20% of verified savings against the customer's default frontier model, net of what routing itself costs the customer (section 2). Shadow mode and the report are free. At $1 million annual spend and 40% savings, that would be $80,000 annual revenue for Tine.

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

As of 2026-11-11 the team's working quality limit for experiments and reports is `epsilon = 0.04`, with `delta = 0.05`. Report results at 0.04 first and keep 0.02 alongside it for comparison. It remains the customer's choice in a deployment, and the rule above still holds: a result must not be made to look better by quietly moving the limit.

Measure cost per completed task, including retries, output length, and caching. Per-token prices alone are insufficient.

Routing has its own cost, and the customer bears it. Everything Gacrux spends to make or check a decision counts toward cost per completed task: cheaper attempts that are escalated, judge-model calls, test or check runs, embedding calls, and compute for models hosted on customer hardware. These calls run on the customer's own provider accounts and infrastructure, like the rest of the deployment, and appear on the customer's bill. Verified savings are therefore net of routing cost: the always-frontier baseline minus what was spent on answers and on routing. A check or judge that costs as much as it saves is not a saving. Record routing cost per request in the ledger, separately from the cost of the answer that was returned, and never report gross savings without it.

## 3. Gacrux architecture

**Read section 12 with this section.** What follows is the design the team started from. Measurements since then changed where the weight sits: predicting the right model from the prompt alone (stages 2 and 3) certifies almost no savings, and checking a cheaper model's answer after it is produced certifies large ones. The stages below remain the intended shape of the system; section 12 says which parts are built, which are measured, and what replaced the prompt classifier as the core.

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

- **Chain:** an ordered list of cheaper models tried before the frontier, each behind a check. Also "try, check, escalate".
- **Check:** what decides whether a cheaper model's answer is accepted: agreement between models, a repository's tests, a judge model, or a combination.
- **Agreement check:** two or three cheaper models answer and the answer is accepted only if they all give the same one. For exact-answer work.
- **Judge (verifier):** a model that reads a request and a cheaper model's answer and accepts or rejects it. Its calls are routing cost.
- **Index model:** the team's name for the step that reads a request before any model answers it (embedding and retrieval). Distinct from the judge, which acts after an answer exists.
- **Frozen policy:** one certified chain for one tenant and task type, written to a file with its evidence (`docs/POLICY.md`).
- **Routing cost:** everything spent to make or check a routing decision; see section 2.
- **Gacrux:** Tine's routing and decision-matrix system.
- **Shadow mode:** sampled parallel model comparisons hidden from the end user.
- **Oracle:** hindsight selection of the cheapest model that succeeded.
- **Best single:** the always-frontier comparison baseline.
- **Tenant:** one customer company, with isolated data and routing state.

## 12. Where things stand (2026-11-16)

This section records measured results and built components. It is a snapshot: check `results/README.md`, which is kept current, before relying on a number here. Everything below is offline evaluation on public benchmark data unless it says otherwise; none of it is a production result.

### What the evidence says

1. **A prompt-only router does not certify savings.** Every representation tried (TF-IDF, sentence embeddings, Qwen3 embeddings, hidden states of a frozen 7B model) with logistic, kNN and cluster routers certifies 0-5% on coding at the working limits with the default rule. The opt-in relative rule reaches 8-9% on `lcb-swe`, and 18% with the net-loss guarantee, but that gain rests on gpt-5's formatting failures on SWE-bench. The official Avengers-Pro code cannot hold a limit. Five routers from the LLMRouter library, run unmodified, either send everything to the frontier or lose 9-47 points of accuracy; they certify under 1% on our pools and on their own benchmark. OpenRouter's auto-router has no quality limit.
2. **Checking the answer certifies large savings.**
   - *Exact-answer work:* agreement between two or three cheap models certifies 32% at the 2% limit and 64% at 4% on MMLU-Pro, and 45-53% at 2% on xRouteBench multiple choice (74% at 4% with pairs, over the limit in 2 of 10 runs), with accuracy at or above the frontier's. Math is weaker (16-20%). It did not repeat on 16 other LLMRouterBench datasets (`results/2026-11-16-agreement-more-datasets`): almost nothing certifies at 2% and only arcc (39%) and math500 (12%) pass 10% at 4%. It needs cheap models close to the frontier in accuracy, about 300 certify tasks for the 4% limit and about 900 for 2% (these are certify tasks only: at the default 40/30/30 split that is about 1,000 and about 3,000 graded tasks in all; `docs/POLICY.md`, Certify on your own graded tasks), and answers that compare as strings.
   - *Code:* free checks (the patch applies, files parse, the repository's existing tests pass) catch about two thirds of wrong patches and certify nothing alone. A strong judge reading what passes (`claude-sonnet-5-5`: rejects 79% of the remaining wrong patches and 9% of right ones, $0.008 per check; `claude-haiku-5-5`: 72% and 9%, $0.0005) gives a chain about a third cheaper than `claude-opus-4.1` and five points more accurate on 274 SWE-bench tasks, net of the judge. This judged chain certifies 22% at a 4% limit when half the tasks certify and nothing at 2%: the limit is the number of graded tasks, not the chain. "Nothing at 2%" is about the judged SWE-bench chain only. Small certified figures on code at 2% exist elsewhere: 3.6% for the response-size check on LiveCodeBench, and 2.3-2.9% for the relative-rule router on `lcb-swe`, which exploits gpt-5's formatting failures.
   - For one chain named in advance and a true loss rate of 0.73%, an 80% chance to certify needs 156 certify tasks at 4%, 257 at 3% and 655 at 2%; with the current many-chain sequence these are lower limits and the measured gap is large (`docs/PLANNING.md`).
   - Agreement between cheap models' patches gives no reliable gain on code: identical patches are too rare (0.5% of pair-tasks), and chains of pairs that accept on similar patches save within about a point of the plain chain either way once a model is not paid twice, and certify no more (`results/2026-11-16-swe-agreement-and-judge`). Similar patches that the judge also accepts are right 90-92% of the time; a chain that needs both certifies in every run at 4% with Sonnet but saves only 3-6% (post hoc).
   - The judges partly remember these public fixes (`results/2026-11-16-judge-probes`): Sonnet writes the merged fix from the issue alone for 41% of tasks. On tasks it cannot recall, a Haiku- or Sonnet-judged chain loses about 2.3% of the frontier's solved tasks, so plan those at a 4-5% limit. `claude-opus-5-5` as the judge, with Haiku screening in front of it, brings that loss to 0.46% at about 24% savings (uncertified, 150 tasks, order chosen after seeing results, Opus's own recall not measured); `docs/JUDGE.md` names it as the check to try first on customer code, not a certified one. Injected approvals, removing the project's name, adding code context and a stricter prompt made no measurable difference (`results/2026-11-17-judge-prompt`). Tiered judges and judges that pick among several patches did not beat the plain chain (`results/2026-11-16-judge-designs`).
   - Open models up to 32B and classifiers trained on graded patches do not work as a judge for code.
3. **The benchmark's SWE-bench grades under-credit patches that need a fuzzy apply.** Prefer grades from our own test run where they exist (`results/2026-11-10-swe-static-checks`).

### What Gacrux is, given that

For each tenant and task type, a frozen policy: cheaper models in order, each behind a check suited to the task (agreement for exact answers; tests then a judge for code), ending at the frontier. The check, not a prompt classifier, is what carries the quality limit. The prompt-reading stages of section 3 are still worth having as a cheap gate in front of a chain (a hidden-state gate raised certified savings from 8% to 15% in one setting), not as the router.

### What is built

| Part | Where | State |
| --- | --- | --- |
| Outcome data, calibration, prompt routers, baselines | `tine/data.py`, `calibration.py`, `routing.py`, `pipeline.py`; `scripts/avengers_pro_baseline.py`, `scripts/llmrouter_baseline.py` | Built and tested; used for the results above |
| Chains with measured or simulated checks, certified by fixed-sequence Clopper-Pearson | `tine/cascade.py`, `tine cascade` | Built and tested; opt-in `--shortlist measured` ranks candidates by their measured checks (mixed result, `results/2026-11-17-measured-shortlist`) |
| Agreement check from recorded answers | `scripts/agreement_pairs.py`; `scripts/agreement_datasets.py` runs it over several datasets | Built and tested |
| Free checks on patches; repository tests without Docker | `scripts/swe_patch_checks.py`, `scripts/swe_repo_tests.py`, Colab notebook | Built; run on 277 of 500 SWE-bench Verified tasks |
| Judge: prompt, one call, spending cap, scoring, join to chains | `tine/judge.py`, `scripts/judge_patches.py` | Built and tested; measured with paid calls ($47.73 for judging, authorised; $71.38 with the live pilot); probes for memory, injection and context in `scripts/judge_probes.py` and `scripts/judge_recall_probe.py` |
| Agreement between patches as a check on code (groups, soft agreement, judge fallback) | `scripts/swe_agreement_checks.py` | Built and tested; no reliable gain |
| Certification planning: tasks needed for a limit, pass probability, split planning | `tine/planning.py`, `scripts/plan_certification.py`, `docs/PLANNING.md` | Built and tested; exact arithmetic for one chain tested alone, no data |
| Frozen policies and a runner with routing cost; answer pattern, `--tenant`, freeze digest (not a signature) and strict guarantee validation | `tine/policy.py`, `tine freeze-policy`, `docs/POLICY.md` | Built and tested. Certifying on a customer's own graded tasks is documented and never done by a customer |
| Proxy with tenant auth, ledger, verified cache, net-of-routing savings | `tine/proxy.py`, `tine/ledger.py`, `docs/PROXY.md` | Development quality; SQLite |
| Ledger audit: where each baseline came from (only proxy-measured ones count as verified), policy digest and stage per routed row, append-only outcome history, opt-in fail-closed ledger (`TINE_LEDGER_REQUIRED`) | `tine/ledger.py`, `tine/proxy.py`, `docs/PROXY.md` | Built and tested with stand-ins; never used by a real customer. No HTTP endpoint for the history. The saving under live routing still cannot be measured: the random frontier holdout is not built |
| Shadow mode, live routing through a policy, named checks (tests behind a webhook, a judge call) | `tine/shadow.py`, `tine/runtime_checks.py`, `tine/proxy.py`, `scripts/shadow_replay.py` | Built and tested with stand-in providers; a replay of recorded answers reproduces the offline estimate. Never run on real traffic |
| Pilot fixes to shadow mode and routing: webhook token rules, answer reader required for agreement route policies, requests with tools skip agreement policies, judged stages skipped for long requests, shadow coverage counts, per-tenant concurrency limit, share-based unknown-cost stop, drift warning against epsilon | `tine/proxy.py`, `tine/shadow.py`, `tine/runtime_checks.py`, `docs/PROXY.md`, `results/2026-11-16-simulated-pilots` | Built and tested with stand-ins; run end to end with real Claude models on public benchmark questions in `results/2026-11-17-live-pilot` (every planned check passed); never used by a real customer. The drift warning compares the share of shadowed requests whose accepted answer differs from the frontier's with epsilon; that share includes questions the frontier got wrong, so it is above the loss and warns on the healthy 2% replay (4.10% differ, 0.99% lost) |
| Savings report page (one self-contained HTML file from the ledger and shadow reports) | `tine/report_page.py`, `scripts/savings_report.py`, `docs/REPORT.md` | Built and tested with stand-in data; shows the freeze digest state, drift warning, shadow coverage and caller-supplied baselines (apart from measured savings), not check rejection reasons; example pages checked in a desktop browser (2026-11-16) before these fields existed and not regenerated since; not checked on a phone or in mail clients; never seen by a customer or made from real traffic |
| Live pilot harness: ask real models for a graded history, then send new requests through the real proxy in shadow mode or routed live, grading every model's reply | `scripts/live_pilot.py`, `results/2026-11-17-live-pilot` | Built, tested with a fake provider, and run once with paid calls ($23.65, authorised) on public benchmark questions; the proxy runs in-process, not as a server |
| Container packaging | `Dockerfile`, `docker-compose.yml`, `deploy/`, `docs/DEPLOY.md` | Written and checked by tests as text; the image has never been built or run |
| Learning loop: certifying a tenant's policy from its own traffic | `docs/LEARNING_LOOP.md` | A design. Not built |
| Evidence brief for outside readers | `docs/EVIDENCE.md` | Written; every number carries its results file |

Not built: nightly per-tenant retraining (a design for certifying a policy from a tenant's own traffic is in `docs/LEARNING_LOOP.md`), distillation, the per-enterprise server beyond the proxy, a production datastore, streaming under a policy, a random frontier holdout to measure savings under live routing.

### Rules learned the hard way

- Report certified numbers first and say when a split, threshold or model list was chosen after seeing a result.
- A pilot on 150 cases overstated the judge (89% against 79% on 1,110). Do not quote pilot numbers as the result.
- Savings are net of routing cost. A check's cost is added to every answer it read before a chain is evaluated.
- Model-written code is executed only on a disposable Linux machine (Colab), never on a team laptop.
- Paid calls need explicit authorisation and a spending cap; keys come from the environment and are never written to the repository or printed.
- Never commit prompts, model answers, per-task outputs, benchmark files or credentials. Results folders hold small summary tables.
