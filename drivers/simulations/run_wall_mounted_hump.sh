#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-"$ROOT/venv/bin/python"}"
BUILD_DIR="${SFEM_BUILD_DIR:-"$ROOT/build64"}"
OUTPUT_DIR="${1:-/private/tmp/sfem_wall_hump}"
GENERATED_DIR="${SFEM_GENERATED_NAVIER_STOKES_DIR:-/private/tmp/sfem_generated_navier_stokes}"

cd "$ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m codegen.framework.materials.navier_stokes \
    --out-dir "$GENERATED_DIR" \
    --element "${SFEM_GENERATED_NS_ELEMENT:-TRI6_TRI3}" \
    --compile \
    --dump-plan

cmake --build "$BUILD_DIR" --target wall_mounted_hump -j "${SFEM_BUILD_JOBS:-4}"

"$BUILD_DIR/wall_mounted_hump" "$OUTPUT_DIR"

"$PYTHON_BIN" drivers/simulations/postprocess_wall_mounted_hump.py \
    "$OUTPUT_DIR" \
    --csv "$OUTPUT_DIR/summary.csv"
