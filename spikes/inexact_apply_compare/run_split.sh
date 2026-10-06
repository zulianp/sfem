#!/usr/bin/env bash
# Generate, build and run the split partial-assembly benchmark.
#
#   run_split.sh [material] [element] [repeats]
#
# element is TET4 (default), HEX8 or TET10.  Keeps a full unfiltered log beside
# the binary and prints progress as it goes, so a slow run is distinguishable
# from a stuck one.
set -euo pipefail

# Mesh ordering.  morton3 by default because that is what SFEM runs:
# `drivers/bench/*` call `smesh::SFC::create_from_env()->reorder`, which defaults
# to it.  hilbert3 and lex are the other orderings worth comparing; random3 is a
# sensitivity check and not a configuration anything produces.
MESH_ORDER="${MESH_ORDER:-morton3}"
# PACKED=1 additionally benchmarks the packed two-pass exact apply, which the
# generator already publishes.  PACK_SIZE sets the elements per pack.
PACKED="${PACKED:-}"
# PACKED_REFERENCE=1 measures the hand-written reference instead of the generated
# kernel -- the two must agree, and that is what makes the port verifiable.
PACKED_REFERENCE="${PACKED_REFERENCE:-}"
PACK_SIZE="${PACK_SIZE:-}"

MATERIAL="${1:-neohookean_ogden}"
ELEMENT="${2:-TET4}"
REPEATS="${3:-5}"
LOWER="$(echo "$ELEMENT" | tr '[:upper:]' '[:lower:]')"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../.." && pwd)"
SFEM="${SFEM_MAIN_CHECKOUT:-$WORKTREE/../sfem}"
# Both spellings are in use across checkouts, so look rather than assume.
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
LOG="$WORK/${MATERIAL}_${LOWER}_${MESH_ORDER}.log"

mkdir -p "$WORK"
echo "START $(date +%T)  material=$MATERIAL element=$ELEMENT repeats=$REPEATS order=$MESH_ORDER" | tee "$LOG"

GEN="$WORK/gen/$MATERIAL/d3/$LOWER"
KGEN="$WORK/gen/$MATERIAL/d3/$KLOWER"
if [ ! -f "$KGEN/${MATERIAL}_${KLOWER}_inexact_apply_inline.hpp" ]; then
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

# Whether the kernels take a state, asked of the generated header rather than
# listed here: a constant tangent -- linear elasticity's -- gathers nothing, so
# the generator gives that kernel neither the connectivity nor the state, and a
# benchmark that passes them does not compile.  The tangent and the exact apply
# are separate questions the generator answers separately, so they are read
# separately even though one material decides both today.
TANGENT_SIGNATURE="$(sed -n "/_inexact_apply_tangent_a_msoa_impl(/,/^) {/p" \
    "$KGEN/${MATERIAL}_${KLOWER}_inexact_apply_inline.hpp" 2>/dev/null || true)"
TAKES_STATE=""
case "$TANGENT_SIGNATURE" in *u_stride*) TAKES_STATE="-DTANGENT_TAKES_STATE";; esac
if [ "$MATERIAL" != "linear_elasticity" ]; then
    TAKES_STATE="$TAKES_STATE -DEXACT_TAKES_STATE"
fi

# HEX8's operator aliases into the PROTEUS_HEX8 translation unit, which pulls
# in mpi.h, so it needs the MPI compiler wrapper and the extra source.
# Every operator includes the mesh types when they are on the path, and those
# reach mpi.h, so the MPI wrapper is needed for all elements -- not only for
# HEX8, which additionally aliases into the PROTEUS_HEX8 translation unit.
EXTRA_TU=""
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
    if command -v "${MPICXX:-mpic++}" >/dev/null 2>&1; then CXX="${MPICXX:-mpic++}"; else CXX="c++"; fi
fi
if [ "$ELEMENT" = "HEX8" ]; then
    for candidate in "$WORK/gen/$MATERIAL/d3/proteus_hex8/${MATERIAL}_proteus_hex8_operator.cpp"; do
        [ -f "$candidate" ] && EXTRA_TU="$candidate"
    done
fi

# OpenMP, without which every `#pragma omp simd` in the generated kernels is a
# no-op.  It was missing, so these benchmarks measured unvectorised code and any
# comparison of a lane-blocked kernel against a scalar one was meaningless --
# the blocking showed up as pure overhead because its pragmas did nothing.
OMPFLAGS="-fopenmp"
if [ "$(uname -s)" = "Darwin" ]; then
    OMPPREFIX="$(brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)"
    OMPFLAGS="-Xpreprocessor -fopenmp -I$OMPPREFIX/include -L$OMPPREFIX/lib -lomp"
fi

# The packed reference, derived from the emitted standard kernel.  Regenerated
# every run: it is a function of the generated header and must not drift from it.
if [ -n "$PACKED_REFERENCE" ]; then
    "$PYTHON" "$HERE/make_packed_reference.py" \
        "$KGEN/${MATERIAL}_${KLOWER}_inexact_apply_inline.hpp" \
        "$KGEN/packed_reference.hpp" "${MATERIAL}_${KLOWER}" 2>&1 | tee -a "$LOG"
fi

# Extra flags for the compile, chiefly -DSIZES to raise the problem sizes: the
# built-in ladder tops out at 64000 elements, which does not fill a 72-core
# Grace socket and so cannot be measured on one.  Read into an array and
# expanded quoted, because -DSIZES={40,64,96,120} unquoted is brace expansion
# and arrives as four separate -DSIZES flags.
read -ra EXTRA_CXXFLAGS <<< "${SFEM_SPIKE_CXXFLAGS:-}"

echo "[2/3] compiling with $CXX ${EXTRA_CXXFLAGS[*]:-}" | tee -a "$LOG"
$CXX -std=c++17 -O3 -march=native -DNDEBUG $TAKES_STATE $OMPFLAGS \
    ${EXTRA_CXXFLAGS[@]+"${EXTRA_CXXFLAGS[@]}"} \
    -DELEMENT_${ELEMENT} \
    ${SHAPE_ORDER:+-DKERNEL_SHAPE_ORDER=$SHAPE_ORDER} \
    -DMESH_ORDER="\"$MESH_ORDER\"" \
    -DMATERIAL_LABEL="\"$MATERIAL\"" \
    -DMATERIAL_INEXACT_HEADER="\"${MATERIAL}_${KLOWER}_inexact_apply_inline.hpp\"" \
    -DEXACT_APPLY=${MATERIAL}_${LOWER}_apply_a_msoa \
    -DTANGENT_KERNEL=${MATERIAL}_${KLOWER}_inexact_apply_tangent_a_msoa_impl \
    -DSTORED_APPLY=${MATERIAL}_${KLOWER}_inexact_apply_stored_a_msoa_impl \
    -DCOMPRESSED_APPLY=${MATERIAL}_${KLOWER}_inexact_apply_compressed_a_msoa_impl \
    ${PACKED:+-DPACKED_EXACT_APPLY=${MATERIAL}_${LOWER}_apply_packed_two_pass_a_msoa} \
    ${PACKED:+-DPACKED_STORED_APPLY=${MATERIAL}_${KLOWER}_inexact_apply_stored_packed_two_pass_a_msoa_impl} \
    ${PACKED_REFERENCE:+-DMATERIAL_PACKED_REFERENCE="\"packed_reference.hpp\""} \
    ${PACKED_REFERENCE:+-DPACKED_REFERENCE_APPLY=${MATERIAL}_${KLOWER}_inexact_apply_stored_packed_two_pass_reference_impl} \
    ${PACK_SIZE:+-DPACK_SIZE=$PACK_SIZE} \
    -o "$WORK/bench_split_${MATERIAL}_${LOWER}_${MESH_ORDER}" \
    "$HERE/bench_split.cpp" "$GEN/${MATERIAL}_${LOWER}_operator.cpp" $EXTRA_TU \
    -I "$GEN" -I "$KGEN" -I "$WORK/gen/$MATERIAL" -I "$WORK/gen/$MATERIAL/d3" \
    -I "$HERE" -I "$WORKTREE/python/codegen/framework/tools" \
    -I "$SFEM/base" -I "$SFEM/algebra" -I "$SFEM/operators" \
    -I "$BUILD" -I "$BUILD/external/smesh" \
    $(find "$SFEM/external/smesh/src" -type d | sed 's/^/-I /') \
    2>&1 | stdbuf -oL tee -a "$LOG"

echo "[3/3] running" | tee -a "$LOG"
"$WORK/bench_split_${MATERIAL}_${LOWER}_${MESH_ORDER}" "$REPEATS" 2>&1 | stdbuf -oL tee -a "$LOG"
echo "END $(date +%T)  full log: $LOG" | tee -a "$LOG"
