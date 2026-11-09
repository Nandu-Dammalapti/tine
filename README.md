# Tine

Tine aims to route AI requests to the cheapest model that meets a calibrated quality requirement, measure savings against a frontier baseline, and learn from each customer's outcomes. The intended product runs in the customer's cloud.

Read [AGENTS.md](AGENTS.md) for the project context, Gacrux routing design, constraints, and build order.

## Current state

The CPU workflow is implemented: validated outcome import, logistic/kNN/cluster baselines, held-out probability and risk calibration, independent test evaluation, and cost-quality reports. There is also an authenticated development proxy with a SQLite ledger, late outcome updates, and a verified exact-answer cache. The proxy defaults to the frontier; it does not automatically deploy an offline routing policy.

The initial setup-only scope in AGENTS.md is superseded by the user's request for this CPU implementation. No LLM weights, benchmark archive, paid API calls, or Colab GPU jobs have been run here. Synthetic fixtures are integration checks, not measured benchmark or customer savings.

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

The cluster method is an Avengers-style heuristic, **not** a claimed reproduction of Avengers-Pro. The oracle is hindsight only. The calibration bound uses a conservative degradation event and a multiple-testing correction; it requires IID representative tasks and cannot promise quality under distribution shift. If evidence is insufficient, the selected policy explicitly falls back to the frontier.

## Prepare original coding tasks

Use the task-only loader when preparing inputs for later model generation and grading. It preserves original splits and check code as text and omits reference solutions. Keep its output separate from graded model outcomes. MBPP here refers to the sanitized subset.

```bash
python -m tine import-coding-tasks --dataset humaneval --input HumanEval.jsonl.gz \
  --output artifacts/humaneval-tasks.jsonl
python -m tine import-coding-tasks --dataset mbpp --input sanitized-mbpp.json \
  --output artifacts/mbpp-tasks.jsonl
```

Pinned source files used to verify the loaders are [OpenAI's HumanEval](https://github.com/openai/human-eval/blob/463c980b59e818ace59f6f9803cd92c749ceae61/data/HumanEval.jsonl.gz) and [Google Research's sanitized MBPP](https://github.com/google-research/google-research/blob/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/sanitized-mbpp.json).

## Exact Colab handoff

Open [the Colab notebook](https://colab.research.google.com/github/Nandu-Dammalapti/tine/blob/main/notebooks/tine_colab.ipynb) and follow [the Colab instructions](docs/COLAB.md). Use CPU and run setup through Section 2 to finish Step 1. Set `FRONTIER` to your intended baseline when prompted. The preflight reports overlapping graded outcomes, split counts, and whether the risk sample could support the default quality bound.

The optional fixed-embedding stage pins a small sentence encoder, checks 50 prompts on CPU first, supports GPU batches, and saves resumable checkpoints on Drive. Its NPZ output plugs directly into `tine run --embeddings`. The archive metadata and the original HumanEval and sanitized MBPP inputs were checked locally. The full graded archive and Colab paths still need a run.

## Development proxy

See [proxy and ledger instructions](docs/PROXY.md). Authenticated OpenAI/Anthropic requests, streaming passthrough, late outcomes, and cache isolation are tested with mocked providers. SQLite is a local development backend. Learned live routing, production Postgres, real provider validation, shadow jobs, nightly retraining, a calibrated deployment process, and model generation/distillation remain future work; none is claimed by the offline report.
