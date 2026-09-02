#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${SFEM_BUILD_DIR:-$ROOT_DIR/build_release}"
TEST_TARGET="${SFEM_TEST_TARGET:-sfem_MooneyRivlinGravityTest}"
OUTPUT_SUBDIR="${SFEM_TEST_OUTPUT_SUBDIR:-test_mooney_rivlin_gravity}"
BIN="$BUILD_DIR/$TEST_TARGET"
HISTORY_MODE=""
HISTORY_STORAGE=""
HISTORY_SCALING="none"
HISTORY_REPLAY=0
OUT_DIR=""

usage() {
    echo "Usage: $0 --history-mode per_qp|per_elem --history-storage float64|float32|float16 [--history-scaling none|tensor|element_prony] [--replay] --out DIR"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --history-mode)
            HISTORY_MODE="$2"
            shift 2
            ;;
        --history-storage)
            HISTORY_STORAGE="$2"
            shift 2
            ;;
        --history-scaling)
            HISTORY_SCALING="$2"
            shift 2
            ;;
        --replay)
            HISTORY_REPLAY=1
            shift
            ;;
        --out)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[error] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$HISTORY_MODE" in
    per_qp) ;;
    per_elem) ;;
    *)
        echo "[error] --history-mode must be per_qp or per_elem" >&2
        exit 2
        ;;
esac

case "$HISTORY_SCALING" in
    none|tensor|element_prony) ;;
    *)
        echo "[error] --history-scaling must be none, tensor, or element_prony" >&2
        exit 2
        ;;
esac

case "$HISTORY_STORAGE" in
    float64) ;;
    float32) ;;
    float16) ;;
    *)
        echo "[error] --history-storage must be float64, float32, or float16" >&2
        exit 2
        ;;
esac

if [[ -z "$OUT_DIR" || "$OUT_DIR" == "/" ]]; then
    echo "[error] --out must name a dedicated output directory" >&2
    exit 2
fi

if [[ ! -x "$BIN" ]]; then
    echo "[error] Missing executable: $BIN" >&2
    echo "[error] Build target $TEST_TARGET first" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR/$OUTPUT_SUBDIR"

(
    cd "$OUT_DIR"
    echo "[info] test=$TEST_TARGET, history_mode=$HISTORY_MODE, history_storage=$HISTORY_STORAGE, history_scaling=$HISTORY_SCALING, replay=$HISTORY_REPLAY"
    SFEM_HISTORY_MODE="$HISTORY_MODE" \
    SFEM_HISTORY_STORAGE="$HISTORY_STORAGE" \
    SFEM_HISTORY_SCALING="$HISTORY_SCALING" \
    SFEM_ENABLE_HISTORY_REPLAY="$HISTORY_REPLAY" \
    SFEM_ENABLE_CONTACT="${SFEM_ENABLE_CONTACT:-0}" \
    SFEM_ENABLE_OUTPUT=1 \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    "$BIN"
) 2>&1 | tee "$OUT_DIR/run.log"
