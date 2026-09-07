#!/usr/bin/env bash
# Build and run the spike bench_op variant, which separates the partial assembly
# from the application.
#
#   run_bench_op_split.sh <element> [resolution] [threads] [material]
#
# The element must match a generated kernel tree (SFEM_SPIKE_GEN).  Links against
# an existing SFEM build (SFEM_BUILD, default build64) rather than adding a
# target to the driver tree, so bench_op and its baseline are untouched.
set -euo pipefail

ELEMENT="${1:-TET4}"
RESOLUTION="${2:-60}"
THREADS="${3:-8}"
MATERIAL="${4:-neohookean_ogden}"
LOWER="$(echo "$ELEMENT" | tr '[:upper:]' '[:lower:]')"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../.." && pwd)"
SFEM="${SFEM_MAIN_CHECKOUT:-$WORKTREE/../sfem}"
BUILD="${SFEM_BUILD:-$SFEM/build64}"
GEN="${SFEM_SPIKE_GEN:?set SFEM_SPIKE_GEN to the generated tree root}"
WORK="${SFEM_SPIKE_WORK:-${TMPDIR:-/tmp}}/bench_op_split"
BIN="$WORK/bench_op_split_${MATERIAL}_${LOWER}"
mkdir -p "$WORK"

# TET10 is reached by promoting a TET4 cube, so the mesh element type and the
# kernel element type differ; the driver is told both.
MESH_ELEM="$ELEMENT"
PROMOTE=0
if [ "$ELEMENT" = "TET10" ]; then MESH_ELEM=TET4; PROMOTE=1; fi

echo "START $(date +%T)  element=$ELEMENT resolution=$RESOLUTION threads=$THREADS"

# Take the include and link lines CMake generated for bench_op, rather than
# reconstructing them: this spike has to see exactly the headers the driver sees,
# and the list is long (ryml, matrix.io, sccd, ssdf, smesh, tbb...).
CXX_INCLUDES=$(sed -n 's/^CXX_INCLUDES = //p' "$BUILD/CMakeFiles/bench_op.dir/flags.make")
LINK_LINE=$(sed -n '1p' "$BUILD/CMakeFiles/bench_op.dir/link.txt" \
            | sed -e 's|^[^ ]*c++ ||' -e 's|CMakeFiles/bench_op.dir/[^ ]*\.o||' \
                  -e 's|-o bench_op||')

( cd "$BUILD" && ${MPICXX:-mpic++} -std=c++17 -O3 -mcpu=apple-m1 -mtune=apple-m1 \
    -Xclang -fopenmp -DNDEBUG \
    -DELEMENT_${ELEMENT} \
    -DMATERIAL_LABEL="\"$MATERIAL\"" \
    -DMATERIAL_INEXACT_HEADER="\"${MATERIAL}_${LOWER}_inexact_apply_inline.hpp\"" \
    -DTANGENT_KERNEL=sfem::codegen::${MATERIAL}_${LOWER}_inexact_apply_tangent_affine_mesh_soa_impl \
    -DSTORED_APPLY=sfem::codegen::${MATERIAL}_${LOWER}_inexact_apply_stored_affine_mesh_soa_impl \
    -DCOMPRESSED_APPLY=sfem::codegen::${MATERIAL}_${LOWER}_inexact_apply_compressed_affine_mesh_soa_impl \
    -I "$GEN/d3/$LOWER" -I "$GEN" -I "$GEN/d3" \
    $CXX_INCLUDES \
    -o "$BIN" "$HERE/bench_op_split.exe.cpp" $LINK_LINE )

echo "[built] $BIN"
SFEM_ELEM_TYPE=$MESH_ELEM SFEM_BASE_RESOLUTION=$RESOLUTION SFEM_PROMOTE_TO_P2=$PROMOTE \
  SFEM_REPEAT=5 OMP_NUM_THREADS=$THREADS OMP_PROC_BIND=true "$BIN"
echo "END $(date +%T)"
