#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${SFEM_BUILD_DIR:-$ROOT_DIR/build_release}"
CASE_RUNNER="$ROOT_DIR/scripts/run_mr_visco_case.sh"
COMPARE="$ROOT_DIR/scripts/compare_mr_visco_history.py"
OUT_BASE="${SFEM_OUT_BASE:-$BUILD_DIR/mr_visco_coupled}"

: "${SFEM_BUILD_JOBS:=12}"
export SFEM_TEST_TARGET=sfem_MooneyRivlinGravityTest
export SFEM_TEST_OUTPUT_SUBDIR=test_mooney_rivlin_gravity
export SFEM_BASE_RESOLUTION="${SFEM_BASE_RESOLUTION:-15}"
export SFEM_DENSITY="${SFEM_DENSITY:-1}"
export SFEM_C10="${SFEM_C10:-800.622}"
export SFEM_C01="${SFEM_C01:-800.108}"
export SFEM_BULK_MODULUS="${SFEM_BULK_MODULUS:-4000}"
export SFEM_DT="${SFEM_DT:-0.005}"
export SFEM_T="${SFEM_T:-3}"
export SFEM_PRONY_G="${SFEM_PRONY_G:-0.4,0.4,0.1,0.05}"
export SFEM_PRONY_TAU="${SFEM_PRONY_TAU:-1,2,5,10}"
export SFEM_USE_WLF="${SFEM_USE_WLF:-0}"
export SFEM_NEWTON_DAMPING="${SFEM_NEWTON_DAMPING:-1}"
export SFEM_ENABLE_CONTACT=0
export SFEM_CONTACT_DIR=0
export SFEM_FIX_SIDE=0
export SFEM_GRAVITY_DIR=1
export SFEM_GRAVITY="${SFEM_GRAVITY:-0.5}"
export SFEM_EXPORT_FREQ="${SFEM_EXPORT_FREQ:-5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

MESH_DIR="$OUT_BASE/mesh"
# Keep previous runs intact, including their mesh and displacement files.
if [[ -e "$OUT_BASE" ]]; then
    echo "[error] Output already exists; choose a new SFEM_OUT_BASE: $OUT_BASE" >&2
    exit 1
fi
python3 -c 'import numpy, yaml, pandas, matplotlib'
mkdir -p "$OUT_BASE"
python3 "$ROOT_DIR/python/sfem/mesh/box_mesh.py" "$MESH_DIR" \
    -c hex8 \
    -x "$((10 * SFEM_BASE_RESOLUTION + 1))" \
    -y "$((SFEM_BASE_RESOLUTION + 1))" \
    -z "$((SFEM_BASE_RESOLUTION + 1))" \
    --width=10 \
    --height=1 \
    --depth=1
export SFEM_MESH="$MESH_DIR"

if [[ "${SFEM_SKIP_BUILD:-0}" != "1" ]]; then
    cmake --build "$BUILD_DIR" --target "$SFEM_TEST_TARGET" --parallel "$SFEM_BUILD_JOBS"
fi

run_case() {
    local mode="$1"
    local storage="$2"
    local scaling="${3:-none}"
    local suffix=""
    case "$scaling" in
        tensor) suffix="_scaled" ;;
        element_prony) suffix="_element_prony" ;;
    esac
    "$CASE_RUNNER" \
        --history-mode "$mode" \
        --history-storage "$storage" \
        --history-scaling "$scaling" \
        --out "$OUT_BASE/cases/${mode}_${storage}${suffix}"
}

run_case per_qp float64
run_case per_qp float32
run_case per_qp float16
run_case per_qp float16 tensor
run_case per_qp float16 element_prony

python3 "$COMPARE" \
    --reference "$OUT_BASE/cases/per_qp_float64" \
    --reference-label per_qp_float64 \
    --candidate "$OUT_BASE/cases/per_qp_float32" \
    --candidate "$OUT_BASE/cases/per_qp_float16" \
    --candidate "$OUT_BASE/cases/per_qp_float16_scaled" \
    --candidate "$OUT_BASE/cases/per_qp_float16_element_prony" \
    --mesh "$MESH_DIR" \
    --end-time "$SFEM_T" \
    --output-subdir "$SFEM_TEST_OUTPUT_SUBDIR" \
    --out "$OUT_BASE/compare"

echo "[info] Cases: $OUT_BASE/cases"
echo "[info] Comparisons: $OUT_BASE/compare"
