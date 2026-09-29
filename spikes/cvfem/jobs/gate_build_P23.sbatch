#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-gate-build-P23
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=gate_build_P23_%j.out
#SBATCH --error=gate_build_P23_%j.err
# Builds for the commit gate, from clean exports of committed trees (no uncommitted edits from
# any session):
#   HEAD 935297062  -> $SCRATCH/cvfem_gate_head/build_gate, every target (drivers, benches, tests),
#                      against SFEM with the zero-start Multigrid API (sfem-omp-zsg)
#   REF  9b480e8d1  -> $SCRATCH/cvfem_gate_ref/build_ref, cvfem_hex8_ns_upwind_bench only, against
#                      the SFEM install that predates it (sfem-omp); the last commit before this
#                      session's performance series, and the reference binary for the packed-
#                      throughput gate.
# Same configuration as build_p14 (Release, mpicxx, trace on) for both, so the A/B differs only
# in the source.
set -u
H=$SCRATCH/cvfem_gate_head
R=$SCRATCH/cvfem_gate_ref
echo "host $(hostname) $(date)"
common="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx -DCVFEM_ENABLE_TRACE=ON"
# The dependency configs (ryml, smesh, matrixio, ...) live in <prefix>/lib64/cmake while SFEM's
# is in lib/cmake, and the spike resolves them relative to SFEM_DIR -- see scripts/alps_env.sh.
# Pass each one explicitly, as build_p14's cache records.
deps() {
    local p=$1 out="-DSFEM_DIR=$1/lib/cmake -DCMAKE_PREFIX_PATH=$1"
    for d in "$p"/lib64/cmake/*/ "$p"/lib/cmake/*/; do
        [ -d "$d" ] || continue
        local n; n=$(basename "$d")
        [ "$n" = "sfem" ] && continue
        out="$out -D${n}_DIR=${d%/}"
    done
    # SCCD and ssdf are not in the OpenMP installs; build_p14 took them from the main prefix.
    for n in SCCD ssdf; do
        [ -d "$p/lib64/cmake/$n" ] || out="$out -D${n}_DIR=$SCRATCH/installations/sfem/lib64/cmake/$n"
    done
    echo "$out"
}

rm -rf "$H/build_gate" "$R/build_ref"
cmake -S "$H" -B "$H/build_gate" $common $(deps $SCRATCH/installations/sfem-omp-zsg) \
    > "$H/configure.log" 2>&1
echo "head configure rc $?"
cmake --build "$H/build_gate" -j 72 > "$H/build.log" 2>&1
echo "head build rc $?"
grep -E "error|Error" "$H/build.log" | head -20

cmake -S "$R" -B "$R/build_ref" $common $(deps $SCRATCH/installations/sfem-omp) \
    > "$R/configure.log" 2>&1
echo "ref configure rc $?"
cmake --build "$R/build_ref" -j 72 --target cvfem_hex8_ns_upwind_bench > "$R/build.log" 2>&1
echo "ref build rc $?"
grep -E "error|Error" "$R/build.log" | head -20

ls -la "$H/build_gate/cvfem_hex8_ns_upwind_bench" "$R/build_ref/cvfem_hex8_ns_upwind_bench"
( cd "$H/build_gate" && ctest -N | tail -1 )
echo "=== DONE $(date)"
