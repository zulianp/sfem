#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${CODE_DIR:-}/merge_git_repos/sfem/venv/bin/activate"

if [[ -n "${CODE_DIR:-}" && -f "$VENV_PATH" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_PATH"
elif [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT_DIR/venv/bin/activate"
fi

cd "$ROOT_DIR"

python -m unittest discover -s python/codegen/framework -p 'test_*.py'
python -m unittest python/codegen/framework/test_neohookean_ogden.py
python -m compileall -q python/codegen/framework
