#!/bin/bash
# Build/run environment for the CVFEM spike on CSCS Alps (daint, GH200).
#
# Source this, or use the helper functions, from a login node:
#     source alps_env.sh && cvfem_configure && cvfem_build
#
# ~/init.sh is the canonical interactive setup, but `uenv start` spawns a subshell
# and so does not compose with a non-interactive `ssh alps '<cmd>'`. This uses
# `uenv run` with the same image instead.

export CVFEM_UENV="${CVFEM_UENV:-prgenv-gnu/24.11:v2}"
export CVFEM_VIEW="${CVFEM_VIEW:-default}"
export SCRATCH="${SCRATCH:-/capstor/scratch/cscs/$USER}"
export CVFEM_SFEM_INSTALL="${CVFEM_SFEM_INSTALL:-$SCRATCH/installations/sfem}"
export CVFEM_ACCOUNT="${CVFEM_ACCOUNT:-c40}"
export CVFEM_SRC="${CVFEM_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export CVFEM_BUILD="${CVFEM_BUILD:-$CVFEM_SRC/build}"

# Two settings are load-bearing and were both found the hard way:
#
#  * -DCMAKE_CXX_COMPILER=mpicxx. Left to its own devices CMake picked a compiler
#    that rejects `_Float16`, and the installed smesh/sfem headers use it, so the
#    build failed inside the *installed* headers rather than in any spike code.
#    mpicxx also supplies the MPI include path that smesh_base.hpp needs.
#
#  * -DCMAKE_PREFIX_PATH=<install root>. The spike's CMakeLists looks for its
#    dependencies under ${SFEM_DIR}/<dep>/<dep>Config.cmake, but on this install
#    ryml, matrixio, smesh, SCCD and ssdf live in lib64/cmake, not lib/cmake.
#
# The uenv version matters too: the installed SFEM::sfem target hardcodes an MPI
# include path from the 24.11:v2 image, so building under v1 fails at generate time
# with "includes non-existent path .../cray-mpich-.../include".

cvfem_uenv() { uenv run --view="$CVFEM_VIEW" "$CVFEM_UENV" -- "$@"; }

# rsync -a preserves source mtimes, which are often older than the object files already
# in the remote build tree -- make then decides everything is up to date and silently
# runs a stale binary. Touch the sources after every sync.
cvfem_touch() { touch "$CVFEM_SRC"/*.hpp "$CVFEM_SRC"/*.cpp "$CVFEM_SRC"/cuda/* 2>/dev/null; }

# The dependency configs are under lib64/cmake/<dep>/ while SFEM's own is under
# lib/cmake/, and the spike's CMakeLists resolves the dependencies relative to SFEM_DIR.
# Pointing SFEM_DIR at lib/cmake therefore finds SFEM and none of ryml, matrixio, smesh,
# SCCD or ssdf, and CMAKE_PREFIX_PATH alone did not rescue it -- configure failed on
# "Could not find a package configuration file provided by ryml". Each is passed
# explicitly, guarded so this still works on an install that puts them elsewhere.
cvfem_dep_dirs() {
    local d args=()
    for d in ryml matrixio smesh SCCD ssdf; do
        if [ -f "$CVFEM_SFEM_INSTALL/lib64/cmake/$d/${d}Config.cmake" ]; then
            args+=("-D${d}_DIR=$CVFEM_SFEM_INSTALL/lib64/cmake/$d")
        fi
    done
    printf '%s\n' "${args[@]}"
}

cvfem_configure() {
    local deps=()
    while IFS= read -r line; do [ -n "$line" ] && deps+=("$line"); done < <(cvfem_dep_dirs)
    cvfem_uenv cmake -S "$CVFEM_SRC" -B "$CVFEM_BUILD" \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DSFEM_DIR="$CVFEM_SFEM_INSTALL/lib/cmake" \
        -DCMAKE_PREFIX_PATH="$CVFEM_SFEM_INSTALL" \
        "${deps[@]}" \
        -DCMAKE_BUILD_TYPE=Release "$@"
}

# Hopper. Separate build tree so the CPU one is not reconfigured with CUDA on.
cvfem_configure_cuda() {
    CVFEM_BUILD="${CVFEM_BUILD_CUDA:-$CVFEM_SRC/build_cuda}" \
        cvfem_configure -DCVFEM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 "$@"
}

cvfem_build_cuda() {
    CVFEM_BUILD="${CVFEM_BUILD_CUDA:-$CVFEM_SRC/build_cuda}" cvfem_build "$@"
}

# The CUDA binaries link libcudart from the uenv image, so they must be launched
# through it -- run directly they fail with "libcudart.so.12: cannot open shared
# object file".
cvfem_run_cuda() {
    srun --account="$CVFEM_ACCOUNT" --partition="${CVFEM_PARTITION:-debug}" \
         --nodes=1 --ntasks=1 --cpus-per-task="${CVFEM_CPUS:-72}" --gpus-per-task=1 \
         --time="${CVFEM_TIME:-00:10:00}" \
         --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" "$@"
}

cvfem_build() {
    cvfem_touch
    cvfem_uenv cmake --build "$CVFEM_BUILD" -j"${BUILD_JOBS:-16}" "$@"
}

# Run on a compute node. The debug partition has a 30 minute limit and is the right
# place for short iteration; use --partition=normal for a real sweep.
cvfem_run() {
    srun --account="$CVFEM_ACCOUNT" --partition="${CVFEM_PARTITION:-debug}" \
         --nodes=1 --ntasks=1 --cpus-per-task="${CVFEM_CPUS:-72}" \
         --time="${CVFEM_TIME:-00:10:00}" \
         --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" "$@"
}
