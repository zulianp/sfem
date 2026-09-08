#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-alloc
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=egal_alloc_%j.out
#SBATCH --error=egal_alloc_%j.err

# Why don't the grid transfers scale?
#
# restrict is 563/565/569/577 us at 8/18/36/72 threads -- flat, and slightly worse with more
# cores. Every element loop in sshex8_restrict and sshex8_prolongate is under `#pragma omp
# for`, so it is not a missing pragma, and W/p + k*p does not fit the numbers either.
#
# What both routines do is malloc their scratch *inside* the parallel region, once per thread
# per call: from_coeffs and to_coeffs, 10 buffers per thread, freed at the end. At 72 threads
# that is 720 malloc/free pairs per call, and the count grows with the thread count while the
# work per thread shrinks -- which is what a flat curve looks like. If glibc is also trimming
# and re-faulting those pages, the cost lands in the kernel, and ~29% of profile samples are
# in unresolved kernel frames.
#
# Two ways to test it without touching smesh:
#   page-faults per run -- allocation churn shows up directly
#   glibc tunables that stop trimming and keep the arena -- if the transfers speed up, the
#   allocator is the cause and the fix is to hoist the scratch out of the per-call path

set -u
# Locate the spike root -- the directory holding CMakeLists.txt, drivers/ and src/.
# Resolved from this script's own path rather than a hardcoded scratch path, so the job
# does not carry one user's $SCRATCH layout and works from any checkout.
SPIKE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SPIKE_ROOT"

UENV_IMAGE="prgenv-gnu/24.11:v2"
UENV_VIEW="default"
BIN="./build_alps/cvfem_hex8_ns_ssgmg"
OUT="$SCRATCH/egal_out_alloc"
mkdir -p "$OUT"

COMMON="SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 SFEM_ELEMENT_REFINE_LEVEL=8 \
        SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_GMG_EGAL=1 SFEM_NL_MAX_IT=1 \
        SFEM_LSOLVE_MAX_IT=60 SFEM_VERIFY_TOL=1e9 OMP_NUM_THREADS=72 \
        OMP_PROC_BIND=close OMP_PLACES=cores"

echo "### baseline: page faults and transfer time ###"
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
    bash -lc "$COMMON perf stat -e page-faults,minor-faults,context-switches \
              stdbuf -oL ${BIN} ${OUT}" 2>&1 \
    | grep -E "top-level phases|restrict|prolong|smooth\[L0\]|page-faults|minor-faults|context-switches|elapsed"
echo

echo "### glibc: no trimming, large top pad (arena kept between calls) ###"
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
    bash -lc "$COMMON GLIBC_TUNABLES=glibc.malloc.trim_threshold=1073741824:glibc.malloc.top_pad=1073741824:glibc.malloc.mmap_threshold=1073741824 \
              perf stat -e page-faults,minor-faults,context-switches \
              stdbuf -oL ${BIN} ${OUT}" 2>&1 \
    | grep -E "top-level phases|restrict|prolong|smooth\[L0\]|page-faults|minor-faults|context-switches|elapsed"
echo

echo "### single-thread transfers for reference (OMP_NUM_THREADS=1) ###"
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
    bash -lc "SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 SFEM_ELEMENT_REFINE_LEVEL=8 \
              SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_GMG_EGAL=1 SFEM_NL_MAX_IT=1 \
              SFEM_LSOLVE_MAX_IT=60 SFEM_VERIFY_TOL=1e9 OMP_NUM_THREADS=1 \
              stdbuf -oL ${BIN} ${OUT}" 2>&1 \
    | grep -E "top-level phases|restrict|prolong|smooth\[L0\]"
