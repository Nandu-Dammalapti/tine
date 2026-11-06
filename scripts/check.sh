#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TINE_DEV_DIR="${TINE_DEV_DIR:-$repo_root/.tine-dev}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TINE_DEV_DIR/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$TINE_DEV_DIR/matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
tine_python="$TINE_DEV_DIR/venv/bin/python"
if [[ ! -x "$tine_python" ]]; then
  echo 'Run bash scripts/setup.sh first (with the same TINE_DEV_DIR, if set).' >&2
  exit 1
fi
UV_CACHE_DIR="$TINE_DEV_DIR/uv-cache" uv pip check --python "$tine_python"
"$TINE_DEV_DIR/venv/bin/ruff" check --isolated --no-cache "$repo_root/scripts/smoke.py"
"$TINE_DEV_DIR/venv/bin/ruff" check --no-cache "$repo_root/tine" "$repo_root/tests"
"$tine_python" -u "$repo_root/scripts/smoke.py"
cd "$repo_root"
"$tine_python" -m pytest -q -p no:cacheprovider
