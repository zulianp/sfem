#!/usr/bin/env bash
# Compile a generated Op with the split turned on and drive it through the Op
# interface, the way a solver would.  Verifies the code SFEM would actually run,
# not the text the generator emitted.
#
#   SFEM_SPIKE_GEN=<generated tree> run_op_inexact_check.sh [resolution]
set -euo pipefail
RESOLUTION="${1:-16}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../.." && pwd)"
SFEM="${SFEM_MAIN_CHECKOUT:-$WORKTREE/../sfem}"
BUILD="${SFEM_BUILD:-$SFEM/build64}"
G="${SFEM_SPIKE_GEN:?set SFEM_SPIKE_GEN to the generated tree root}"
WORK="${SFEM_SPIKE_WORK:-${TMPDIR:-/tmp}}/op_inexact_check"
mkdir -p "$WORK"

CXX_INCLUDES=$(sed -n 's/^CXX_INCLUDES = //p' "$BUILD/CMakeFiles/bench_op.dir/flags.make")
LINK_LINE=$(sed -n '1p' "$BUILD/CMakeFiles/bench_op.dir/link.txt" \
            | sed -e 's|^[^ ]*c++ ||' -e 's|CMakeFiles/bench_op.dir/[^ ]*\.o||' -e 's|-o bench_op||')

( cd "$BUILD" && ${MPICXX:-mpic++} -std=c++17 -O2 -DNDEBUG -Xclang -fopenmp \
    -o "$WORK/op_inexact_check" "$HERE/op_inexact_check.exe.cpp" \
    $(ls "$G"/op/*.cpp) $(ls "$G"/d3/*/*.cpp) \
    -I "$G/op" -I "$G" -I "$G/d3" $(find "$G/d3" -type d | sed 's/^/-I /') \
    $CXX_INCLUDES $LINK_LINE )

echo "[built] $WORK/op_inexact_check"
SFEM_BASE_RESOLUTION=$RESOLUTION "$WORK/op_inexact_check"
