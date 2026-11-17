# Tine

Tine aims to route AI requests to the cheapest model that meets a calibrated quality requirement, measure savings against a frontier baseline, and learn from each customer's outcomes. The intended product runs in the customer's cloud.

Read [AGENTS.md](AGENTS.md) for the project context, Gacrux routing design, constraints, and build order.

## Current state

The offline toolkit is implemented: validated outcome import, logistic/kNN/cluster routers, held-out calibration, independent test evaluation, cost-quality reports, escalation chains with measured or simulated checks (`tine cascade`), agreement and judge checks, and frozen policies (`tine freeze-policy`). There is also an authenticated development proxy with a SQLite ledger, late outcome updates and a verified exact-answer cache. The proxy sends every request to the frontier unless a frozen policy is configured; it can then run that policy in shadow mode (report only) or serve answers from it. Shadow mode and policy routing are tested with stand-in providers and a replay of recorded answers only. Nothing has served real traffic. See AGENTS.md section 12 for the current state.

The initial setup-only scope in AGENTS.md was widened by later requests. Since then the LLMRouterBench and xRouteBench archives have been imported, Colab jobs have run (embeddings, hidden states, repository tests, an open-model judge), and authorised paid runs called Claude models: as a judge ($47.73: $12.69 in `results/2026-11-14-api-judge`, $31.02 in `results/2026-11-16-judge-probes` and $4.02 in `results/2026-11-17-judge-prompt`), and to answer real requests through the proxy in two live pilots ($23.65 in `results/2026-11-17-live-pilot` and $60.00 from a separate credits key in `results/2026-11-17-live-pilot-more`), $131.38 in all. The probes found that the judges partly remember these public fixes (Sonnet recalls 41% of them); on the tasks they cannot recall, a Haiku- or Sonnet-judged chain loses about 2.3% of the frontier's solved tasks, and 0.46% with `claude-opus-5-5` as the judge (uncertified, 150 tasks). Apart from the live pilots, all results are on recorded answers from public benchmarks; the live pilot used public benchmark questions too, not customer traffic. Synthetic fixtures are integration checks, not measured benchmark or customer savings.

## Results

Start with the index of all results, [results/README.md](results/README.md), and the one-page comparison, [results/SCOREBOARD.md](results/SCOREBOARD.md). A short summary with a source for every number: [docs/EVIDENCE.md](docs/EVIDENCE.md).
First coding benchmark findings: [results/2026-11-09-coding-benchmark](results/2026-11-09-coding-benchmark/README.md).
Predictor-gated escalation chains: [results/2026-11-10-gated-chains](results/2026-11-10-gated-chains/README.md).
Outside research: [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md).
Avengers-Pro head-to-head: [results/2026-11-10-avengers-pro](results/2026-11-10-avengers-pro/README.md) and [docs/AVENGERS_PRO.md](docs/AVENGERS_PRO.md).
Static patch checks on SWE-bench: [results/2026-11-10-swe-static-checks](results/2026-11-10-swe-static-checks/README.md).
Judge check pilot: [results/2026-11-11-judge-pilot](results/2026-11-11-judge-pilot/README.md) and [docs/JUDGE.md](docs/JUDGE.md).
Agreement check on exact-answer tasks: [results/2026-11-11-agreement-check](results/2026-11-11-agreement-check/README.md).
Router rule and guarantee options: [results/2026-11-11-router-rules](results/2026-11-11-router-rules/README.md). Architecture status: [docs/ARCHITECTURE_STATUS.md](docs/ARCHITECTURE_STATUS.md).
Response check for escalation chains: [results/2026-11-10-response-check](results/2026-11-10-response-check/README.md).
Hidden-state representations: [results/2026-11-11-hidden-states](results/2026-11-11-hidden-states/README.md). Trained verifier: [results/2026-11-13-trained-verifier](results/2026-11-13-trained-verifier/README.md).
Agreement on a second benchmark and LLMRouter baselines: [results/2026-11-14-xroutebench](results/2026-11-14-xroutebench/README.md).
Paid judge and the first code chains that certify: [results/2026-11-14-api-judge](results/2026-11-14-api-judge/README.md).
Shadow mode and live routing replayed: [results/2026-11-16-shadow-replay](results/2026-11-16-shadow-replay/README.md).
Agreement on more exact-answer datasets: [results/2026-11-16-agreement-more-datasets](results/2026-11-16-agreement-more-datasets/README.md). Agreement between patches on code: [results/2026-11-16-swe-agreement-and-judge](results/2026-11-16-swe-agreement-and-judge/README.md).

## Documentation

- [docs/HANDOFF.md](docs/HANDOFF.md): handoff note (2026-11-17): where things stand, what is proven and not, open problems and what to build next.
- [docs/EVIDENCE.md](docs/EVIDENCE.md): evidence brief for outside readers, with a source for every number.
- [docs/PROXY.md](docs/PROXY.md): the development proxy, ledger, shadow mode, routing through a policy and named checks.
- [docs/POLICY.md](docs/POLICY.md): frozen policies.
- [docs/REPORT.md](docs/REPORT.md): the savings report page. `scripts/savings_report.py` writes a tenant's savings report as one HTML file.
- [docs/DEPLOY.md](docs/DEPLOY.md): container packaging. The image was built and run once from an earlier Dockerfile (40 paid requests, 2026-11-17); the current files were built and self-tested once the same day (18 of 18 checks, a stand-in provider; `results/2026-11-17-container-selftest`) and have never been deployed or used by a customer.
- [docs/PLANNING.md](docs/PLANNING.md): how many graded tasks certification needs.
- [docs/LEARNING_LOOP.md](docs/LEARNING_LOOP.md): a design for certifying a tenant's policy from its own traffic. Not built.
- [docs/CHECKS.md](docs/CHECKS.md), [docs/SWE_CHECKS.md](docs/SWE_CHECKS.md), [docs/JUDGE.md](docs/JUDGE.md): checks for chains.

## Setup

Prerequisites: Python 3.12 with `venv`, [uv](https://docs.astral.sh/uv/getting-started/installation/), Bash, and Git. The existing cloud environment also provides Node.js/npm and build tools; these are optional for the current Python workflow.

From the repository root:

```bash
bash scripts/setup.sh
source .tine-dev/venv/bin/activate
bash scripts/check.sh
```

The setup script installs the exact versions in `requirements-dev.lock`, including NumPy, pandas, SciPy, scikit-learn, Matplotlib, FastAPI, Uvicorn, HTTPX, pytest, and Ruff. It can be rerun without changing the lockfile. Run commands from the repository root. For an installed CLI, optionally use `uv pip install --python .tine-dev/venv/bin/python --no-deps -e .` and run `tine --help`. `requirements-dev.in` records the direct tool choices. Keep lock updates deliberate and rerun the checks after changing dependencies.

Local environments, caches, credentials, raw data, models, and generated reports are excluded from Git.

## Existing cloud environment

The prepared cloud instance retains its tools outside the checkout. Reuse them with:

```bash
export TINE_DEV_DIR=/workspace/.tine-dev
bash scripts/setup.sh
source "$TINE_DEV_DIR/venv/bin/activate"
bash scripts/check.sh
```

Each cloud task is isolated. Use its existing checkout; create a worktree only when explicitly requested. Cloud environment publication remains a separate action in environment settings.

## What the check proves

The check runs the application test suite, lint, dependency compatibility, and the original 50-row environment smoke check. Tests cover data validation, source import, safe archive extraction, tenant isolation, routing, calibration, train/test separation, interrupted embedding resume, and the development proxy/ledger with mocked providers.

The environment-check chart is at `$TINE_DEV_DIR/synthetic-cost-smoke.png` (default `.tine-dev/synthetic-cost-smoke.png`). Passing tests do not establish real routing savings or a production quality guarantee.

## Run the complete CPU experiment

```bash
python -m tine demo --output artifacts/demo --tasks 2000
```

This trains the CPU baselines on explicitly synthetic data, selects thresholds on held-out calibration rows, evaluates a separate test set, and writes `report.md`, `metrics.csv`, `calibration.json`, `splits.csv`, `decisions.csv`, `curves.csv`, and `cost-quality.png`. Costs used for routing are estimated from training rows; observed test costs are used only to evaluate decisions.

Add `--latency-samples 100` to `demo` or `run` to measure warm single-request CPU routing on up to 100 distinct held-out tasks per tenant and method. The benchmark uses the fitted router and frozen calibration policies, with three untimed warm-up calls per method. It adds sample counts and p50/p99 milliseconds to `metrics.csv`, and writes individual timings, selected model names, and hardware context to `routing-latency.json`. Timings include task validation, feature transformation or lookup, calibrated scoring, cost estimation, and policy selection. They exclude DataFrame construction, embedding generation, the verified-answer cache, network, and provider calls. These are descriptive sample percentiles, not production latency guarantees. The default `--latency-samples 0` keeps this benchmark disabled; batch-average prediction timing is reported separately.

Latency sampling supports both `--rule absolute` and `--rule relative`. Relative sampling includes comparing each model's score with the frontier's score inside the timed decision. The latency artifact records the rule and loss setting, and each observation uses the same frozen policy as the batch evaluation.

Experiment outputs are generated in a temporary sibling directory and published together when the run succeeds. Reruns preserve the previous results on generation failure and keep unrelated files such as user notes. A hidden sibling `.tine.lock` file coordinates writers targeting the same directory on Linux and macOS. Choose a dedicated output directory outside the current working directory's ancestor chain. Output directories and generated artifact paths must be regular directories/files. If publication rollback fails, the error identifies the preserved recovery directory. Demo runs also publish their source outcome snapshot with the reports. Canonical CSV/JSONL exports use atomic file replacement to preserve existing data when encoding or writing fails. Reports recognize synthetic provenance even when those outcomes are reloaded through `tine run` or mixed with recorded data.

For real data, import existing graded outcomes rather than regenerating answers:

```bash
python -m tine import-llmrouterbench --input /path/to/bench \
  --dataset livecodebench --run-selection latest --output artifacts/outcomes.jsonl
python -m tine inspect --input artifacts/outcomes.jsonl
python -m tine preflight --input artifacts/outcomes.jsonl --frontier EXACT_MODEL_NAME
python -m tine run --input artifacts/outcomes.jsonl --frontier EXACT_MODEL_NAME \
  --epsilon 0.02 --delta 0.05 --output artifacts/real-evaluation
```

`EXACT_MODEL_NAME` is your predeclared baseline from the printed model names. Canonical CSV/JSONL rows contain tenant, task ID, prompt, task type, model, a known/pending success label, and recorded per-task cost. Optional token counts, latency, and timestamps remain unknown when absent. Missing outcomes are never silently treated as failures; incomplete tasks and duplicate prompts are excluded and recorded. Import does not execute model-generated code.

Measurements, routing scores, and cost matrices must contain real numbers. Boolean and complex-valued measurements are rejected before conversion, including direct Python calls to routing, calibration, and evaluation. Success labels remain Boolean or pending.

Outcome tables require unique column names. Outcome and check CSV imports reject duplicate headers, malformed quoted fields, and records with more or fewer fields than the header. Keep an explicit empty cell (including its delimiter) for an unknown value; quoted commas and multiline text are supported. JSON/JSONL imports reject duplicate object fields instead of silently choosing one value. Direct router calls also reject ambiguous thresholds, non-Boolean fallback switches, and Boolean embedding values.

When supplying embeddings, `summary.json` records `embedding_sha256` alongside the outcome dataset fingerprint. It identifies the validated float64 feature values and task/prompt identities used in the experiment. Reordering or recompressing an embedding archive, or including unused rows, leaves this fingerprint unchanged; changing used vectors changes it. This identifies feature data, not the embedding model's training history.

The cluster method is an Avengers-style heuristic, **not** a claimed reproduction of Avengers-Pro. The oracle is hindsight only. The default `--loss degradation` bounds the event that the frontier succeeds and the selected model fails. Optional `--loss net` bounds overall accuracy loss using a degradation upper bound minus a gain lower bound; individual tasks may regress even when overall accuracy improves. Both use multiple-testing corrections, require IID representative tasks, and cannot promise quality under distribution shift. If evidence is insufficient, the selected policy explicitly falls back to the frontier.

Use `python -m tine cascade` to evaluate try-cheap-first escalation policies on existing outcomes. See [measured check evidence](docs/CHECKS.md) for the `--checks` schema, tenant isolation, missing-check behavior, and the limits of simulated check results. Cascade reports publish together and retain synthetic labels. `tine cascade` adds no check cost of its own. For a result net of routing cost, the check's cost must already be in the outcomes it reads: `scripts/agreement_pairs.py` charges every member of a group, and `scripts/judge_patches.py checks` adds each judge's recorded cost to the answers it read. Test-run compute and check latency are not costed anywhere yet.

`python -m tine combine-outcomes` can append fresh outcome files to an existing pool while preserving `source_task_type` and reporting every source. Missing source labels use another recorded label for the same tenant/task, or that row's current task type, before `--task-type` relabels the pool. Conflicting source labels for one task are rejected. Source names such as `NA` remain literal text when reloading CSV files.

## Prepare original coding tasks

Use the task-only loader when preparing inputs for later model generation and grading. It preserves original splits and check code as text and omits reference solutions. Keep its output separate from graded model outcomes. MBPP here refers to the sanitized subset.

Task JSON must have unique field names; duplicate IDs or check fields within an object are rejected rather than silently replaced. Task exports use atomic file replacement, preserving an existing output if encoding or publication fails. Output-file symlinks are rejected.

```bash
python -m tine import-coding-tasks --dataset humaneval --input HumanEval.jsonl.gz \
  --output artifacts/humaneval-tasks.jsonl
python -m tine import-coding-tasks --dataset mbpp --input sanitized-mbpp.json \
  --output artifacts/mbpp-tasks.jsonl
```

Pinned source files used to verify the loaders are [OpenAI's HumanEval](https://github.com/openai/human-eval/blob/463c980b59e818ace59f6f9803cd92c749ceae61/data/HumanEval.jsonl.gz) and [Google Research's sanitized MBPP](https://github.com/google-research/google-research/blob/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/sanitized-mbpp.json).

## Exact Colab handoff

Open [the Colab notebook](https://colab.research.google.com/github/Nandu-Dammalapti/tine/blob/main/notebooks/tine_colab.ipynb) and follow [the Colab instructions](docs/COLAB.md). Use CPU and run setup through Section 2 to finish Step 1. Set `FRONTIER` to your intended baseline when prompted. The preflight reports overlapping graded outcomes, split counts, and whether the risk sample could support the default quality bound.

The optional fixed-embedding stage pins a small sentence encoder, checks 50 prompts on CPU first, supports GPU batches, and saves resumable checkpoints on Drive. Its NPZ output plugs directly into `tine run --embeddings`. The archive metadata and the original HumanEval and sanitized MBPP inputs were checked locally. The archive download, import and the Colab embedding path have since been run (`results/2026-11-09-coding-benchmark`).

## Development proxy

See [proxy and ledger instructions](docs/PROXY.md). Authenticated OpenAI/Anthropic requests, streaming passthrough, late outcomes, cache isolation, shadow mode and routing through a frozen policy (with named checks: a tests webhook and a judge call) are tested with mocked providers. Shadow mode and routing through agreement policies have also run with real current models on public benchmark questions (`results/2026-11-17-live-pilot`, `results/2026-11-17-live-pilot-more`); the named checks are tested with stand-ins only. SQLite is a local development backend. None of this has served a customer's traffic. Production Postgres, streaming under a policy, nightly retraining and model generation/distillation remain future work; none is claimed by the offline report. Container packaging is in [docs/DEPLOY.md](docs/DEPLOY.md). The image was built and run once from an earlier Dockerfile (40 paid requests, 2026-11-17); the current files were built and self-tested once the same day (18 of 18 checks, a stand-in provider; `results/2026-11-17-container-selftest`) and have never been deployed or used by a customer.

The ledger records answer-generation and routing/checking costs separately. Verified savings subtract both; requests with unknown routing costs are excluded from that comparison. With a route policy the proxy records routing cost itself for requests the policy handled. Otherwise supply measured `routing_cost_usd` through the outcome endpoint, including for cache hits. Existing records retain unknown routing costs after migration.

The ledger also records where each baseline came from (only baselines the proxy measured count as verified), the policy file digest and stage behind each routed answer, and an append-only history of outcome updates. `TINE_LEDGER_REQUIRED` makes the proxy return 503 rather than serve an answer it could not record. Under live routing the saving cannot be measured without a random share of requests sent to the frontier, which is not built.
