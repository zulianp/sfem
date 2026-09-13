#!/usr/bin/env bash
# Generate, build and run the two-unit split benchmark.
#
#   run_mixed.sh [element] [repeats]
#
# element is TET4 (default), HEX8 or TET10.  Mooney-Rivlin elasticity plus
# Kelvin-Voigt viscosity is one material with two units, so this generates one
# tree and links two sets of kernels out of it: an energy unit whose exact
# entry point is `apply_a_msoa`, and a residual unit whose exact entry point is
# `jacobian_action_a_msoa`.  That asymmetry is why this cannot just be
# `run_split.sh` with a different material name.
#
# Keeps a full unfiltered log beside the binary and prints progress as it goes,
# so a slow run is distinguishable from a stuck one.
set -euo pipefail

MATERIAL=mooney_rivlin_kelvin_voigt_newmark
ELEMENT="${1:-TET4}"
REPEATS="${2:-5}"
LOWER="$(echo "$ELEMENT" | tr '[:upper:]' '[:lower:]')"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../.." && pwd)"
SFEM="${SFEM_MAIN_CHECKOUT:-$WORKTREE/../sfem}"
PYTHON="${SFEM_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for candidate in "$SFEM/.venv/bin/python" "$SFEM/venv/bin/python" \
                     "$WORKTREE/.venv/bin/python" "$WORKTREE/venv/bin/python"; do
        [ -x "$candidate" ] && PYTHON="$candidate" && break
    done
fi
: "${PYTHON:?set SFEM_PYTHON: no venv found beside the checkout}"
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
BUILD="${SFEM_BUILD:-$SFEM/build}"
WORK="${SFEM_SPIKE_WORK:-${TMPDIR:-/tmp}}/inexact_apply_split"
LOG="$WORK/mixed_${LOWER}.log"

ELASTIC="${MATERIAL}_elastic"
VISCOUS="${MATERIAL}_viscous"

mkdir -p "$WORK"
echo "START $(date +%T)  mixed element=$ELEMENT repeats=$REPEATS" | tee "$LOG"

GEN="$WORK/gen/$MATERIAL/d3/$LOWER"
KGEN="$WORK/gen/$MATERIAL/d3/$KLOWER"
if [ ! -f "$KGEN/${ELASTIC}_${KLOWER}_inexact_apply_inline.hpp" ]; then
    echo "[1/3] generating kernels" | tee -a "$LOG"
    ( cd "$WORKTREE" && PYTHONPATH=python \
        "$PYTHON" - "$WORK/gen" "$MATERIAL" "$ELEMENT" <<'PY'
import dataclasses, sys
out, name, element = sys.argv[1], sys.argv[2], sys.argv[3]
from sfem import gen
from codegen.framework.materials.mooney_rivlin_kelvin_voigt_newmark import material
gen.generate(dataclasses.replace(material, inexact_apply=True),
             "%s/%s" % (out, name), elements=(element,), clean=False)
PY
    ) 2>&1 | stdbuf -oL tee -a "$LOG"
else
    echo "[1/3] kernels already generated, reusing" | tee -a "$LOG"
fi

# Every operator includes the mesh types when they are on the path, and those
# reach mpi.h, so the MPI wrapper is needed for all elements -- not only for
# HEX8, which additionally aliases into the PROTEUS_HEX8 translation unit and
# so needs its sources as well.
# The two units do not always produce one operator source each.  Where an
# element forwards to its Cartesian twin, both units' wrappers land in a single
# forwarding source and `_collapse_duplicate_operators` keeps one copy of it --
# so link the operator sources that exist rather than naming two.
UNIT_TU=""
for candidate in "$GEN"/*_operator.cpp; do
    case "$candidate" in *_inexact_apply_operator.cpp) continue ;; esac
    [ -f "$candidate" ] && UNIT_TU="$UNIT_TU $candidate"
done

EXTRA_TU=""
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
    if command -v "${MPICXX:-mpic++}" >/dev/null 2>&1; then CXX="${MPICXX:-mpic++}"; else CXX="c++"; fi
fi
if [ "$ELEMENT" = "HEX8" ]; then
    for unit in "$ELASTIC" "$VISCOUS"; do
        candidate="$WORK/gen/$MATERIAL/d3/proteus_hex8/${unit}_proteus_hex8_operator.cpp"
        [ -f "$candidate" ] && EXTRA_TU="$EXTRA_TU $candidate"
    done
fi

echo "[2/3] compiling with $CXX" | tee -a "$LOG"
$CXX -std=c++17 -O3 -march=native -DNDEBUG \
    -DELEMENT_${ELEMENT} \
    ${SHAPE_ORDER:+-DKERNEL_SHAPE_ORDER=$SHAPE_ORDER} \
    -DELASTIC_INEXACT_HEADER="\"${ELASTIC}_${KLOWER}_inexact_apply_inline.hpp\"" \
    -DVISCOUS_INEXACT_HEADER="\"${VISCOUS}_${KLOWER}_inexact_apply_inline.hpp\"" \
    -DEXACT_ELASTIC_APPLY=${ELASTIC}_${LOWER}_apply_a_msoa \
    -DEXACT_VISCOUS_ACTION=${VISCOUS}_${LOWER}_jacobian_action_a_msoa \
    -DELASTIC_TANGENT=sfem::codegen::${ELASTIC}_${KLOWER}_inexact_apply_tangent_a_msoa_impl \
    -DELASTIC_STORED=sfem::codegen::${ELASTIC}_${KLOWER}_inexact_apply_stored_a_msoa_impl \
    -DELASTIC_COMPRESS=sfem::codegen::${ELASTIC}_${KLOWER}_inexact_apply_compressed_a_msoa_impl \
    -DVISCOUS_TANGENT=sfem::codegen::${VISCOUS}_${KLOWER}_inexact_apply_tangent_a_msoa_impl \
    -DVISCOUS_STORED=sfem::codegen::${VISCOUS}_${KLOWER}_inexact_apply_stored_a_msoa_impl \
    -DVISCOUS_COMPRESS=sfem::codegen::${VISCOUS}_${KLOWER}_inexact_apply_compressed_a_msoa_impl \
    -DPACKED_STORED_APPLY \
    -DELASTIC_PACKED_STORED=sfem::codegen::${ELASTIC}_${KLOWER}_inexact_apply_stored_packed_two_pass_a_msoa_impl \
    -DVISCOUS_PACKED_STORED=sfem::codegen::${VISCOUS}_${KLOWER}_inexact_apply_stored_packed_two_pass_a_msoa_impl \
    -o "$WORK/bench_mixed_${LOWER}" \
    "$HERE/bench_mixed.cpp" $UNIT_TU $EXTRA_TU \
    -I "$GEN" -I "$WORK/gen/$MATERIAL" -I "$WORK/gen/$MATERIAL/d3" \
    -I "$HERE" -I "$WORKTREE/python/codegen/framework/tools" \
    -I "$SFEM/base" -I "$SFEM/algebra" -I "$SFEM/operators" \
    -I "$BUILD" -I "$BUILD/external/smesh" \
    $(find "$SFEM/external/smesh/src" -type d | sed 's/^/-I /') \
    ${MIXED_EXTRA_FLAGS:-} \
    2>&1 | stdbuf -oL tee -a "$LOG"

echo "[3/3] running" | tee -a "$LOG"
"$WORK/bench_mixed_${LOWER}" "$REPEATS" 2>&1 | stdbuf -oL tee -a "$LOG"
echo "END $(date +%T)  full log: $LOG" | tee -a "$LOG"
