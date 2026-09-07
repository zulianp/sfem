#!/usr/bin/env bash
# Warp-severity sweep: how the projected apply deviates as the displacement is
# warped away from a constant-gradient (linear) one.
#
#   run_warp.sh [material] [element] [n]
set -euo pipefail
MATERIAL="${1:-neohookean_ogden}"; ELEMENT="${2:-TET10}"; N="${3:-16}"
LOWER="$(echo "$ELEMENT" | tr '[:upper:]' '[:lower:]')"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../.." && pwd)"
SFEM="${SFEM_MAIN_CHECKOUT:-$WORKTREE/../sfem}"
BUILD="${SFEM_BUILD:-$SFEM/build}"
WORK="${SFEM_SPIKE_WORK:-${TMPDIR:-/tmp}}/inexact_apply_split"
GEN="$WORK/gen/$MATERIAL/d3/$LOWER"
LOG="$WORK/warp_${MATERIAL}_${LOWER}.log"
mkdir -p "$WORK"
TAKES_STATE="-DEXACT_TAKES_STATE"
[ "$MATERIAL" = "linear_elasticity" ] && TAKES_STATE=""
EXTRA=""; CXX="${CXX:-c++}"
if [ "$ELEMENT" = "HEX8" ]; then
    CXX="${MPICXX:-mpic++}"
    EXTRA="$WORK/gen/$MATERIAL/d3/proteus_hex8/${MATERIAL}_proteus_hex8_operator.cpp"
fi
echo "START $(date +%T) warp sweep $MATERIAL $ELEMENT n=$N" | tee "$LOG"
$CXX -std=c++17 -O2 -DNDEBUG $TAKES_STATE -DELEMENT_${ELEMENT} ${WARP_EXTRA_FLAGS:-} \
    -DMATERIAL_LABEL="\"$MATERIAL\"" \
    -DMATERIAL_INEXACT_HEADER="\"${MATERIAL}_${LOWER}_inexact_apply_inline.hpp\"" \
    -DEXACT_APPLY=${MATERIAL}_${LOWER}_apply_affine_mesh_soa \
    -DTANGENT_KERNEL=${MATERIAL}_${LOWER}_inexact_apply_tangent_affine_mesh_soa_impl \
    -DSTORED_APPLY=${MATERIAL}_${LOWER}_inexact_apply_stored_affine_mesh_soa_impl \
    -o "$WORK/warp_${MATERIAL}_${LOWER}" "$HERE/warp_sweep.cpp" \
    "$GEN/${MATERIAL}_${LOWER}_operator.cpp" $EXTRA \
    -I "$HERE" -I "$GEN" -I "$WORK/gen/$MATERIAL" -I "$WORK/gen/$MATERIAL/d3" \
    -I "$SFEM/base" -I "$SFEM/algebra" -I "$SFEM/operators" \
    -I "$BUILD" -I "$BUILD/external/smesh" \
    $(find "$SFEM/external/smesh/src" -type d | sed 's/^/-I /') 2>&1 | stdbuf -oL tee -a "$LOG"
"$WORK/warp_${MATERIAL}_${LOWER}" "$N" 2>&1 | stdbuf -oL tee -a "$LOG"
echo "END $(date +%T)" | tee -a "$LOG"
