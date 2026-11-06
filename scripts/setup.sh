#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tine_dev_dir="${TINE_DEV_DIR:-$repo_root/.tine-dev}"
tine_python="${TINE_PYTHON:-python3}"
command -v uv >/dev/null || { echo 'Install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
"$tine_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required; select it with TINE_PYTHON"'

mkdir -p "$tine_dev_dir/cache" "$tine_dev_dir/matplotlib"
if [[ ! -x "$tine_dev_dir/venv/bin/python" ]]; then
  "$tine_python" -m venv "$tine_dev_dir/venv"
fi
"$tine_dev_dir/venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing environment must use Python 3.12"'
UV_CACHE_DIR="$tine_dev_dir/uv-cache" uv pip install \
  --python "$tine_dev_dir/venv/bin/python" \
  --requirement "$repo_root/requirements-dev.lock"
UV_CACHE_DIR="$tine_dev_dir/uv-cache" uv pip check \
  --python "$tine_dev_dir/venv/bin/python"
printf 'Environment prepared at %s/venv\n' "$tine_dev_dir"
