#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/scripts"
BUILD_DIR="$ROOT_DIR/build_release"
BIN_DIR="$BUILD_DIR"
OUT_BASE="${SFEM_OUT_BASE:-$BUILD_DIR}"

mkdir -p "$OUT_BASE"

echo "[info] ROOT_DIR=$ROOT_DIR"
echo "[info] BIN_DIR=$BIN_DIR"
echo "[info] OUT_BASE=$OUT_BASE"

cd "$BUILD_DIR"

if [[ -z "${SFEM_BUILD_CMD:-}" ]]; then
    echo "[info] Building with: make -j12"
    make -j12
else
    echo "[info] Building with: $SFEM_BUILD_CMD"
    eval "$SFEM_BUILD_CMD"
fi

if [[ ! -x "$BIN_DIR/sfem_MRViscoExcelValidation" ]]; then
    echo "[error] Missing executable: $BIN_DIR/sfem_MRViscoExcelValidation"
    exit 1
fi

if [[ ! -x "$BIN_DIR/sfem_MooneyRivlinGravityTest" ]]; then
    echo "[error] Missing executable: $BIN_DIR/sfem_MooneyRivlinGravityTest"
    exit 1
fi

: "${USE_CUBE:=1}"
: "${SFEM_ENABLE_CONTACT:=1}"
: "${SFEM_CONTACT_DIR:=2}"
: "${SFEM_FIX_SIDE:=0}"
: "${SFEM_GRAVITY_DIR:=2}"
: "${SFEM_GRAVITY:=10}"
: "${SFEM_OBSTACLE_TYPE:=0}"
: "${SFEM_CONTACT_PLANE:=0}"
: "${SFEM_DENSITY:=1}"
: "${SFEM_C10:=8.622}"
: "${SFEM_C01:=8.108}"
: "${SFEM_BULK_MODULUS:=40.0}"
: "${SFEM_DT:=0.01}"
: "${SFEM_T:=3}"
: "${SFEM_ENABLE_OUTPUT:=1}"
: "${SFEM_EXPORT_FREQ:=5}"
# Setting B (strong relaxation)
: "${SFEM_PRONY_G:=0.45, 0.45, 0.01, 0.05}"
: "${SFEM_PRONY_TAU:=0.1, 0.5, 1.5, 3}"
: "${SFEM_WLF_T_REF:=-54.29}"
: "${SFEM_TEMPERATURE:=20}"
: "${SFEM_WLF_C1:=16.6263}"
: "${SFEM_WLF_C2:=47.4781}"
: "${SFEM_USE_WLF:=0}"
: "${SFEM_INIT_GAP:=0.00}"
: "${SFEM_BASE_RESOLUTION:=30}"
: "${SFEM_HEMISPHERE_CENTER_X:=1}"
: "${SFEM_HEMISPHERE_CENTER_Y:=0.5}"
: "${SFEM_HEMISPHERE_CENTER_Z:=-0.8}"
: "${SFEM_HEMISPHERE_RADIUS:=0.8}"

run_case() {
    local label="$1"
    local mode="$2"
    local out_dir="$OUT_BASE/$label"

    mkdir -p "$out_dir"

    if [[ -n "$mode" ]]; then
        export SFEM_HISTORY_MODE="$mode"
    else
        unset SFEM_HISTORY_MODE
    fi

    echo "[info] Running case: $label (SFEM_HISTORY_MODE=${SFEM_HISTORY_MODE:-<unset>})"

    USE_CUBE="$USE_CUBE" \
    SFEM_ENABLE_CONTACT="$SFEM_ENABLE_CONTACT" \
    SFEM_CONTACT_DIR="$SFEM_CONTACT_DIR" \
    SFEM_FIX_SIDE="$SFEM_FIX_SIDE" \
    SFEM_GRAVITY_DIR="$SFEM_GRAVITY_DIR" \
    SFEM_GRAVITY="$SFEM_GRAVITY" \
    SFEM_OBSTACLE_TYPE="$SFEM_OBSTACLE_TYPE" \
    SFEM_CONTACT_PLANE="$SFEM_CONTACT_PLANE" \
    SFEM_DENSITY="$SFEM_DENSITY" \
    SFEM_C10="$SFEM_C10" \
    SFEM_C01="$SFEM_C01" \
    SFEM_BULK_MODULUS="$SFEM_BULK_MODULUS" \
    SFEM_DT="$SFEM_DT" \
    SFEM_T="$SFEM_T" \
    SFEM_ENABLE_OUTPUT="$SFEM_ENABLE_OUTPUT" \
    SFEM_EXPORT_FREQ="$SFEM_EXPORT_FREQ" \
    SFEM_PRONY_G="$SFEM_PRONY_G" \
    SFEM_PRONY_TAU="$SFEM_PRONY_TAU" \
    SFEM_WLF_T_REF="$SFEM_WLF_T_REF" \
    SFEM_TEMPERATURE="$SFEM_TEMPERATURE" \
    SFEM_WLF_C1="$SFEM_WLF_C1" \
    SFEM_WLF_C2="$SFEM_WLF_C2" \
    SFEM_USE_WLF="$SFEM_USE_WLF" \
    SFEM_INIT_GAP="$SFEM_INIT_GAP" \
    SFEM_BASE_RESOLUTION="$SFEM_BASE_RESOLUTION" \
    SFEM_HEMISPHERE_CENTER_X="$SFEM_HEMISPHERE_CENTER_X" \
    SFEM_HEMISPHERE_CENTER_Y="$SFEM_HEMISPHERE_CENTER_Y" \
    SFEM_HEMISPHERE_CENTER_Z="$SFEM_HEMISPHERE_CENTER_Z" \
    SFEM_HEMISPHERE_RADIUS="$SFEM_HEMISPHERE_RADIUS" \
    "$BIN_DIR/sfem_MRViscoExcelValidation" 2>&1 | tee "$out_dir/visco_excel.log"

    USE_CUBE="$USE_CUBE" \
    SFEM_ENABLE_CONTACT="$SFEM_ENABLE_CONTACT" \
    SFEM_CONTACT_DIR="$SFEM_CONTACT_DIR" \
    SFEM_FIX_SIDE="$SFEM_FIX_SIDE" \
    SFEM_GRAVITY_DIR="$SFEM_GRAVITY_DIR" \
    SFEM_GRAVITY="$SFEM_GRAVITY" \
    SFEM_OBSTACLE_TYPE="$SFEM_OBSTACLE_TYPE" \
    SFEM_CONTACT_PLANE="$SFEM_CONTACT_PLANE" \
    SFEM_DENSITY="$SFEM_DENSITY" \
    SFEM_C10="$SFEM_C10" \
    SFEM_C01="$SFEM_C01" \
    SFEM_BULK_MODULUS="$SFEM_BULK_MODULUS" \
    SFEM_DT="$SFEM_DT" \
    SFEM_T="$SFEM_T" \
    SFEM_ENABLE_OUTPUT="$SFEM_ENABLE_OUTPUT" \
    SFEM_EXPORT_FREQ="$SFEM_EXPORT_FREQ" \
    SFEM_PRONY_G="$SFEM_PRONY_G" \
    SFEM_PRONY_TAU="$SFEM_PRONY_TAU" \
    SFEM_WLF_T_REF="$SFEM_WLF_T_REF" \
    SFEM_TEMPERATURE="$SFEM_TEMPERATURE" \
    SFEM_WLF_C1="$SFEM_WLF_C1" \
    SFEM_WLF_C2="$SFEM_WLF_C2" \
    SFEM_USE_WLF="$SFEM_USE_WLF" \
    SFEM_INIT_GAP="$SFEM_INIT_GAP" \
    SFEM_BASE_RESOLUTION="$SFEM_BASE_RESOLUTION" \
    SFEM_HEMISPHERE_CENTER_X="$SFEM_HEMISPHERE_CENTER_X" \
    SFEM_HEMISPHERE_CENTER_Y="$SFEM_HEMISPHERE_CENTER_Y" \
    SFEM_HEMISPHERE_CENTER_Z="$SFEM_HEMISPHERE_CENTER_Z" \
    SFEM_HEMISPHERE_RADIUS="$SFEM_HEMISPHERE_RADIUS" \
    "$BIN_DIR/sfem_MooneyRivlinGravityTest" 2>&1 | tee "$out_dir/gravity.log"

    cp -f visco_validation_results.csv "$out_dir/" 2>/dev/null || true
    cp -f visco_validation_*.png "$out_dir/" 2>/dev/null || true
    cp -f visco_validation_*.pdf "$out_dir/" 2>/dev/null || true
    cp -f plot_visco_validation.py "$out_dir/" 2>/dev/null || true
    if [[ -d "test_mooney_rivlin_gravity" ]]; then
        rm -rf "$out_dir/test_mooney_rivlin_gravity"
        cp -R "test_mooney_rivlin_gravity" "$out_dir/"
    fi
}

run_case "baseline_per_qp" ""
run_case "per_elem" "per_elem"

python "$SCRIPT_DIR/compare_mr_visco_history.py" \
    --baseline "$OUT_BASE/baseline_per_qp" \
    --per-elem "$OUT_BASE/per_elem" \
    --out "$OUT_BASE/compare"

echo "[info] Done. Results in: $OUT_BASE"
