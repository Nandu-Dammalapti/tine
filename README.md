# Tine

Tine aims to route AI requests to the cheapest model that meets a calibrated quality requirement, measure savings against a frontier baseline, and learn from each customer's outcomes. The intended product runs in the customer's cloud.

Read [AGENTS.md](AGENTS.md) for the project context, Gacrux routing design, constraints, and build order.

## Current state

This repository contains development prerequisites and an environment smoke check. The proxy, router, calibration pipeline, storage, and dashboard have not been implemented. Setup does not download models or datasets, train models, call provider APIs, or start services. No API keys, GPU, Docker daemon, or database is required.

## Setup

Prerequisites: Python 3.12 with `venv`, [uv](https://docs.astral.sh/uv/getting-started/installation/), Bash, and Git. The existing cloud environment also provides Node.js/npm and build tools; these are optional for the current Python workflow.

From the repository root:

```bash
bash scripts/setup.sh
source .tine-dev/venv/bin/activate
bash scripts/check.sh
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

The check processes 50 synthetic CSV rows, checks numeric cost totals and pairwise distances, computes a binomial confidence bound, renders a PNG with a headless plotting backend, and exercises in-memory FastAPI/HTTPX request handling. It also checks dependency compatibility and tool startup.

The generated chart is at `$TINE_DEV_DIR/synthetic-cost-smoke.png` (default `.tine-dev/synthetic-cost-smoke.png`). These checks verify the installed tools. They are not Tine application tests, routing benchmarks, or evidence of a quality guarantee. There is no application test suite yet.

## Scope of the next work

Start with small CPU fixtures and existing graded outcomes. Add dataset loaders and schemas, baselines, held-out calibration, and cost-quality evaluation before model generation or infrastructure expansion. Follow the detailed build order and constraints in AGENTS.md. Keep prompts and outcomes isolated by tenant, and measure cost per completed task.
