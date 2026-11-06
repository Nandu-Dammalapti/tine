# Tine

Tine aims to route AI requests to the cheapest model that meets a calibrated quality requirement, measure savings against a frontier baseline, and learn from each customer's outcomes. The intended product runs in the customer's cloud.

Read [AGENTS.md](AGENTS.md) for the project context, Gacrux routing design, constraints, and build order.

## Current state

`tine/` holds the Gacrux router v1 evaluation path on CPU: loaders, per-model success predictors, a τ certified on held-out data, baselines and cost-quality charts. `colab/` holds the GPU work (prompt embeddings, open-model answer generation). The proxy, logger, shadow mode, storage and dashboard are not built yet. See [NEXT_STEPS.md](NEXT_STEPS.md) for status and what to do next.

So far the pipeline has run on invented data only. Setup does not download models or datasets, call provider APIs, or start services.

## Setup

Prerequisites: Python 3.12 with `venv`, [uv](https://docs.astral.sh/uv/getting-started/installation/), Bash, and Git. The existing cloud environment also provides Node.js/npm and build tools; these are optional for the current Python workflow.

From the repository root:

```bash
bash scripts/setup.sh
source .tine-dev/venv/bin/activate
bash scripts/check.sh
python -m tine demo
```

The setup script installs the exact versions in `requirements-dev.lock`, including NumPy, pandas, SciPy, scikit-learn, Matplotlib, FastAPI, Uvicorn, HTTPX, pytest, and Ruff. It can be rerun without changing the lockfile. `requirements-dev.in` records the direct tool choices. Keep lock updates deliberate and rerun the checks after changing dependencies.

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

`scripts/check.sh` checks dependency compatibility, lints, runs the environment smoke check, and runs the test suite in `tests/` (schema, loaders, routing rule, τ certification including a simulation of its failure rate, baselines, and an end-to-end run on synthetic data). Passing tests show the code does what it says on invented data; they are not benchmark results.

## Scope of the next work

Start with small CPU fixtures and existing graded outcomes. Add dataset loaders and schemas, baselines, held-out calibration, and cost-quality evaluation before model generation or infrastructure expansion. Follow the detailed build order and constraints in AGENTS.md. Keep prompts and outcomes isolated by tenant, and measure cost per completed task.
