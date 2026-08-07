#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/python/codegen/framework/twophaseflow/generated}"
shift || true

source "${CODE_DIR:?CODE_DIR must be set}/merge_git_repos/sfem/venv/bin/activate"

cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR/python${PYTHONPATH:+:$PYTHONPATH}" \
python -m codegen.framework.materials.two_phase_flow \
    --out-dir "$OUT_DIR" \
    --compile \
    "$@"
