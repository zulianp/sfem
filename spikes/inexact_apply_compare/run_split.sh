#!/usr/bin/env bash
# Generate, build and run the split partial-assembly benchmark.
#
#   run_split.sh [material] [element] [repeats]
#
# element is TET4 (default), HEX8 or TET10.  Keeps a full unfiltered log beside
# the binary and prints progress as it goes, so a slow run is distinguishable
# from a stuck one.
set -euo pipefail

MATERIAL="${1:-neohookean_ogden}"
ELEMENT="${2:-TET4}"
REPEATS="${3:-5}"
LOWER="$(echo "$ELEMENT" | tr '[:upper:]' '[:lower:]')"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../.." && pwd)"
SFEM="${SFEM_MAIN_CHECKOUT:-$WORKTREE/../sfem}"
PYTHON="${SFEM_PYTHON:-$SFEM/.venv/bin/python}"
BUILD="${SFEM_BUILD:-$SFEM/build}"
WORK="${SFEM_SPIKE_WORK:-${TMPDIR:-/tmp}}/inexact_apply_split"
LOG="$WORK/${MATERIAL}_${LOWER}.log"

mkdir -p "$WORK"
echo "START $(date +%T)  material=$MATERIAL element=$ELEMENT repeats=$REPEATS" | tee "$LOG"

GEN="$WORK/gen/$MATERIAL/d3/$LOWER"
if [ ! -f "$GEN/${MATERIAL}_${LOWER}_inexact_apply_inline.hpp" ]; then
    echo "[1/3] generating kernels (slow for HEX8)" | tee -a "$LOG"
    ( cd "$WORKTREE" && PYTHONPATH=python:python/codegen/framework/materials \
        "$PYTHON" - "$WORK/gen" "$MATERIAL" "$ELEMENT" <<'PY'
import dataclasses, sys
out, name, element = sys.argv[1], sys.argv[2], sys.argv[3]
from sfem import gen
module = __import__(name)
gen.generate(dataclasses.replace(module.material, inexact_apply=True),
             "%s/%s" % (out, name), elements=(element,), clean=False)
PY
    ) 2>&1 | stdbuf -oL tee -a "$LOG"
else
    echo "[1/3] kernels already generated, reusing" | tee -a "$LOG"
fi

# linear elasticity's exact apply takes no state: its tangent is constant.
TAKES_STATE="-DEXACT_TAKES_STATE"
if [ "$MATERIAL" = "linear_elasticity" ]; then TAKES_STATE=""; fi

# HEX8's operator aliases into the PROTEUS_HEX8 translation unit, which pulls
# in mpi.h, so it needs the MPI compiler wrapper and the extra source.
EXTRA_TU=""
CXX="${CXX:-c++}"
if [ "$ELEMENT" = "HEX8" ]; then
    CXX="${MPICXX:-mpic++}"
    for candidate in "$WORK/gen/$MATERIAL/d3/proteus_hex8/${MATERIAL}_proteus_hex8_operator.cpp"; do
        [ -f "$candidate" ] && EXTRA_TU="$candidate"
    done
fi

echo "[2/3] compiling with $CXX" | tee -a "$LOG"
$CXX -std=c++17 -O3 -march=native -DNDEBUG $TAKES_STATE \
    -DELEMENT_${ELEMENT} \
    -DMATERIAL_LABEL="\"$MATERIAL\"" \
    -DMATERIAL_INEXACT_HEADER="\"${MATERIAL}_${LOWER}_inexact_apply_inline.hpp\"" \
    -DEXACT_APPLY=${MATERIAL}_${LOWER}_apply_affine_mesh_soa \
    -DTANGENT_KERNEL=${MATERIAL}_${LOWER}_inexact_apply_tangent_affine_mesh_soa_impl \
    -DSTORED_APPLY=${MATERIAL}_${LOWER}_inexact_apply_stored_affine_mesh_soa_impl \
    -DCOMPRESSED_APPLY=${MATERIAL}_${LOWER}_inexact_apply_compressed_affine_mesh_soa_impl \
    -o "$WORK/bench_split_${MATERIAL}_${LOWER}" \
    "$HERE/bench_split.cpp" "$GEN/${MATERIAL}_${LOWER}_operator.cpp" $EXTRA_TU \
    -I "$GEN" -I "$WORK/gen/$MATERIAL" -I "$WORK/gen/$MATERIAL/d3" \
    -I "$SFEM/base" -I "$SFEM/algebra" -I "$SFEM/operators" \
    -I "$BUILD" -I "$BUILD/external/smesh" \
    $(find "$SFEM/external/smesh/src" -type d | sed 's/^/-I /') \
    2>&1 | stdbuf -oL tee -a "$LOG"

echo "[3/3] running" | tee -a "$LOG"
"$WORK/bench_split_${MATERIAL}_${LOWER}" "$REPEATS" 2>&1 | stdbuf -oL tee -a "$LOG"
echo "END $(date +%T)  full log: $LOG" | tee -a "$LOG"
