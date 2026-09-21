#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VENV_PATH="${CODE_DIR:-}/merge_git_repos/sfem/venv/bin/activate"

OUT_DIR="${1:-$SCRIPT_DIR/generated/mooney_rivlin}"
ELEMENT_TYPE="${2:-HEX8}"
VECTOR_SIZE="${3:-16}"

if [[ -n "${CODE_DIR:-}" && -f "$VENV_PATH" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_PATH"
elif [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT_DIR/venv/bin/activate"
fi

export PYTHONPATH="$ROOT_DIR/python${PYTHONPATH:+:$PYTHONPATH}"

python -m codegen.framework.materials.mooney_rivlin \
    --out-dir "$OUT_DIR" \
    --element "$ELEMENT_TYPE" \
    --vector-size "$VECTOR_SIZE" \
    --compile
