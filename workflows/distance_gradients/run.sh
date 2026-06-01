#!/usr/bin/env bash

set -e
set -x

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

SFEM_DISTANCE_GRADIENT_OUTPUT="${SFEM_DISTANCE_GRADIENT_OUTPUT:-distance_gradients}"
SFEM_DISTANCE_GRADIENT_STEPS="${SFEM_DISTANCE_GRADIENT_STEPS:-8}"

rm -rf "$SFEM_DISTANCE_GRADIENT_OUTPUT"

TEST_BIN="${SFEM_DISTANCE_GRADIENT_TEST:-}"
if [[ -z "$TEST_BIN" && -x "$ROOT/build64/sfem_DistanceGradientsTest" ]]; then
    TEST_BIN="$ROOT/build64/sfem_DistanceGradientsTest"
fi

if [[ -z "$TEST_BIN" && -x "$ROOT/build/sfem_DistanceGradientsTest" ]]; then
    TEST_BIN="$ROOT/build/sfem_DistanceGradientsTest"
fi

if [[ -z "$TEST_BIN" ]]; then
    TEST_BIN="$(command -v sfem_DistanceGradientsTest || true)"
fi

if [[ -z "$TEST_BIN" ]]; then
    echo "Could not find sfem_DistanceGradientsTest. Set SFEM_DISTANCE_GRADIENT_TEST=/path/to/sfem_DistanceGradientsTest." >&2
    exit 1
fi

SFEM_DISTANCE_GRADIENT_OUTPUT="$SFEM_DISTANCE_GRADIENT_OUTPUT" \
SFEM_DISTANCE_GRADIENT_STEPS="$SFEM_DISTANCE_GRADIENT_STEPS" \
    $LAUNCH "$TEST_BIN"

python3 "$HERE/write_xdmf.py" "$SFEM_DISTANCE_GRADIENT_OUTPUT" "$((SFEM_DISTANCE_GRADIENT_STEPS + 1))"
