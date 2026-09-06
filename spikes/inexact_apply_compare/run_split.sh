#!/usr/bin/env bash
# Generate, build and run the split partial-assembly benchmark.
#
#   run_split.sh [material] [repeats]
#
# Keeps a full unfiltered log beside the binary and prints progress as it goes,
# so a slow run is distinguishable from a stuck one.
set -euo pipefail

MATERIAL="${1:-neohookean_ogden}"
REPEATS="${2:-5}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../.." && pwd)"
SFEM="${SFEM_MAIN_CHECKOUT:-$WORKTREE/../sfem}"
PYTHON="${SFEM_PYTHON:-$SFEM/.venv/bin/python}"
BUILD="${SFEM_BUILD:-$SFEM/build}"
WORK="${SFEM_SPIKE_WORK:-${TMPDIR:-/tmp}}/inexact_apply_split"
LOG="$WORK/$MATERIAL.log"

mkdir -p "$WORK"
echo "START $(date +%T)  material=$MATERIAL repeats=$REPEATS" | tee "$LOG"

echo "[1/3] generating kernels" | tee -a "$LOG"
( cd "$WORKTREE" && PYTHONPATH=python:python/codegen/framework/materials \
    "$PYTHON" - "$WORK/gen" "$MATERIAL" <<'PY'
import dataclasses, sys
out, name = sys.argv[1], sys.argv[2]
from sfem import gen
module = __import__(name)
gen.generate(dataclasses.replace(module.material, inexact_apply=True),
             "%s/%s" % (out, name), elements=("TET4",), clean=True)
PY
) 2>&1 | stdbuf -oL tee -a "$LOG"

GEN="$WORK/gen/$MATERIAL/d3/tet4"
# linear elasticity's exact apply takes no state: its tangent is constant.
TAKES_STATE="-DEXACT_TAKES_STATE"
if [ "$MATERIAL" = "linear_elasticity" ]; then TAKES_STATE=""; fi

echo "[2/3] compiling" | tee -a "$LOG"
c++ -std=c++17 -O3 -march=native -DNDEBUG $TAKES_STATE \
    -DMATERIAL_LABEL="\"$MATERIAL\"" \
    -DMATERIAL_INEXACT_HEADER="\"${MATERIAL}_tet4_inexact_apply_inline.hpp\"" \
    -DEXACT_APPLY=${MATERIAL}_tet4_apply_affine_mesh_soa \
    -DFUSED_APPLY=${MATERIAL}_tet4_apply_inexact_affine_mesh_soa_impl \
    -DTANGENT_KERNEL=${MATERIAL}_tet4_inexact_apply_tangent_affine_mesh_soa_impl \
    -DSTORED_APPLY=${MATERIAL}_tet4_inexact_apply_stored_affine_mesh_soa_impl \
    -DCOMPRESSED_APPLY=${MATERIAL}_tet4_inexact_apply_compressed_affine_mesh_soa_impl \
    -o "$WORK/bench_split_$MATERIAL" \
    "$HERE/bench_split.cpp" "$GEN/${MATERIAL}_tet4_operator.cpp" \
    -I "$GEN" -I "$WORK/gen/$MATERIAL" -I "$WORK/gen/$MATERIAL/d3" \
    -I "$SFEM/base" -I "$SFEM/algebra" -I "$SFEM/operators" \
    -I "$BUILD" -I "$BUILD/external/smesh" \
    $(find "$SFEM/external/smesh/src" -type d | sed 's/^/-I /') \
    2>&1 | stdbuf -oL tee -a "$LOG"

echo "[3/3] running" | tee -a "$LOG"
"$WORK/bench_split_$MATERIAL" "$REPEATS" 2>&1 | stdbuf -oL tee -a "$LOG"
echo "END $(date +%T)  full log: $LOG" | tee -a "$LOG"
