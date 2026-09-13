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
. "$HERE/kernel_element.sh"
# HEX8 and QUAD4 forward to their lexicographic twin, so the templated bodies
# this benchmark instantiates are the twin's while the C ABI symbols it calls
# stay this element's.  KLOWER names the first, LOWER the second.
KERNEL_ELEMENT="$(kernel_element "$ELEMENT")"
KLOWER="$(echo "$KERNEL_ELEMENT" | tr '[:upper:]' '[:lower:]')"
# The permutation that goes with it.  These benchmarks call the `..._impl`
# templates directly, so they bypass the generated forwarder and have to reorder
# the connectivity themselves; `element_mesh.inc` does it from this.
SHAPE_ORDER="$(kernel_shape_order "$ELEMENT")"
WORK="${SFEM_SPIKE_WORK:-${TMPDIR:-/tmp}}/inexact_apply_split"
GEN="$WORK/gen/$MATERIAL/d3/$LOWER"
KGEN="$WORK/gen/$MATERIAL/d3/$KLOWER"
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
    ${SHAPE_ORDER:+-DKERNEL_SHAPE_ORDER=$SHAPE_ORDER} \
    -DMATERIAL_LABEL="\"$MATERIAL\"" \
    -DMATERIAL_INEXACT_HEADER="\"${MATERIAL}_${KLOWER}_inexact_apply_inline.hpp\"" \
    -DEXACT_APPLY=${MATERIAL}_${LOWER}_apply_a_msoa \
    -DTANGENT_KERNEL=${MATERIAL}_${KLOWER}_inexact_apply_tangent_a_msoa_impl \
    -DSTORED_APPLY=${MATERIAL}_${KLOWER}_inexact_apply_stored_a_msoa_impl \
    -o "$WORK/warp_${MATERIAL}_${LOWER}" "$HERE/warp_sweep.cpp" \
    "$GEN/${MATERIAL}_${LOWER}_operator.cpp" $EXTRA \
    -I "$HERE" -I "$GEN" -I "$WORK/gen/$MATERIAL" -I "$WORK/gen/$MATERIAL/d3" \
    -I "$SFEM/base" -I "$SFEM/algebra" -I "$SFEM/operators" \
    -I "$BUILD" -I "$BUILD/external/smesh" \
    $(find "$SFEM/external/smesh/src" -type d | sed 's/^/-I /') 2>&1 | stdbuf -oL tee -a "$LOG"
"$WORK/warp_${MATERIAL}_${LOWER}" "$N" 2>&1 | stdbuf -oL tee -a "$LOG"
echo "END $(date +%T)" | tee -a "$LOG"
