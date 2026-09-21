#!/usr/bin/env bash
# The generated Laplacian against SFEM's own, on one GPU.
#
# This one links against SFEM's headers rather than the standalone shim: the
# hand-written kernel is part of the library and pulls in `sfem_base.hpp`,
# `sfem_cuda_base.hpp` and the smesh frontend, so it takes the same include
# flags the library is built with.
set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SFEM=$HOME/sfem-wt
FLAGS="$SFEM/build/CMakeFiles/sfem.dir/flags.make"
INCLUDES=$(grep -m1 "^CXX_INCLUDES" "$FLAGS" | cut -d= -f2-)
DEFINES=$(grep -m1 "^CXX_DEFINES" "$FLAGS" | cut -d= -f2-)
# nvcc is not the MPI wrapper, and SFEM's base header includes `mpi.h`
MPI_INC="-I$(dirname "$(find /user-environment -name mpi.h 2>/dev/null | head -1)")"
NVCC="nvcc -std=c++17 -O3 -arch=sm_90 -DNDEBUG -diag-suppress 177"
mkdir -p "$ROOT/hw"

$NVCC $DEFINES $INCLUDES $MPI_INC -I "$SFEM/operators/tet4/cuda" -I "$SFEM/operators/cuda" \
    -c "$SFEM/operators/tet4/cuda/cu_tet4_laplacian.cu" -o "$ROOT/hw/cu_tet4_laplacian.o" \
    > "$ROOT/hw/handwritten.log" 2>&1 || { echo "FAIL hand-written laplacian"; grep -m5 error "$ROOT/hw/handwritten.log"; exit 1; }

$NVCC $DEFINES $INCLUDES $MPI_INC -I "$SFEM/operators/tet4/cuda" -I "$SFEM/operators/cuda" \
    -c "$SFEM/operators/tet4/cuda/cu_tet4_linear_elasticity.cu" \
    -o "$ROOT/hw/cu_tet4_linear_elasticity.o" \
    > "$ROOT/hw/handwritten_le.log" 2>&1 || { echo "FAIL hand-written elasticity"; grep -m5 error "$ROOT/hw/handwritten_le.log"; exit 1; }

$NVCC $DEFINES $INCLUDES $MPI_INC -I "$ROOT/cuda/linear_elasticity" \
    -c "$ROOT/cuda/linear_elasticity/d3/tet4/linear_elasticity_tet4_operator.cu" \
    -o "$ROOT/hw/linear_elasticity_tet4_operator.o" \
    > "$ROOT/hw/generated_le.log" 2>&1 || { echo "FAIL generated elasticity"; grep -m5 error "$ROOT/hw/generated_le.log"; exit 1; }

# With SFEM's includes, not standalone: the generated header takes its scalar
# and index types from `sfem_base.hpp` when it can see one, and SFEM's `idx_t`
# is `int` where a standalone generation falls back to `ptrdiff_t`.  Compiling
# the two halves against different definitions of `idx_t` is a silent ABI
# mismatch -- the kernel read the driver's `int` indices as `ptrdiff_t` and
# faulted on the first element.
$NVCC $DEFINES $INCLUDES $MPI_INC -I "$ROOT/cuda/laplace" \
    -c "$ROOT/cuda/laplace/d3/tet4/laplace_tet4_operator.cu" \
    -o "$ROOT/hw/laplace_tet4_operator.o" > "$ROOT/hw/generated.log" 2>&1 \
    || { echo "FAIL generated kernel"; grep -m5 error "$ROOT/hw/generated.log"; exit 1; }

$NVCC $DEFINES $INCLUDES $MPI_INC -I "$SFEM/operators/tet4/cuda" -I "$SFEM/operators/cuda" \
    -c "$ROOT/handwritten_compare.cu" -o "$ROOT/hw/driver.o" \
    > "$ROOT/hw/driver.log" 2>&1 || { echo "FAIL driver"; grep -m6 error "$ROOT/hw/driver.log"; exit 1; }

# `sfem_abort` is the hand-written kernel's error path and calls `MPI_Abort`
mpicxx -std=c++17 -O3 -DNDEBUG $DEFINES $INCLUDES -c "$SFEM/base/sfem_base.cpp" \
    -o "$ROOT/hw/sfem_base.o" > "$ROOT/hw/base.log" 2>&1 \
    || { echo "FAIL sfem_base"; grep -m4 error "$ROOT/hw/base.log"; exit 1; }
MPI_LIB=$(dirname "$(find /user-environment -name 'libmpi.so*' 2>/dev/null | head -1)")

nvcc -O3 -arch=sm_90 -o "$ROOT/handwritten_compare" \
    "$ROOT/hw/driver.o" "$ROOT/hw/cu_tet4_laplacian.o" "$ROOT/hw/laplace_tet4_operator.o" \
    "$ROOT/hw/cu_tet4_linear_elasticity.o" "$ROOT/hw/linear_elasticity_tet4_operator.o" \
    "$ROOT/hw/sfem_base.o" -L"$MPI_LIB" -lmpi \
    > "$ROOT/hw/link.log" 2>&1 || { echo "FAIL link"; grep -m8 -E "undefined|error" "$ROOT/hw/link.log"; exit 1; }
echo "built handwritten_compare"
