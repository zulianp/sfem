#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-egal-threads
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=egal_threads_%j.out
#SBATCH --error=egal_threads_%j.err

# Is the thread clamp throttling the fine level?
#
# thread_clamped gives each level ceil(ndofs / SFEM_GMG_DOFS_PER_THREAD) threads, capped at
# the machine's. At the default 20000 and 242,500 dofs the *fine* level gets 12 of 72 cores,
# and level 1 gets one. perf agrees: 85.5e9 cycles over 2.065 s on 72 cores is ~17% of what
# the machine could retire, and 12/72 is 16.7%.
#
# The clamp is not wrong in principle -- it was added because an 81-node level cost 156 us on
# one thread and 14.2 ms on eight, ninety times worse for having more cores. The question is
# whether 20000 dofs per thread is the right threshold, since it was set from the coarse
# levels' behaviour and applies to every level.
#
# Fixed work: one Newton step, a fixed number of V-cycles, so every arm does the same
# arithmetic and only the thread mapping changes.

set -u
cd "$SCRATCH/sfem/spikes/cvfem"

UENV_IMAGE="prgenv-gnu/24.11:v2"
UENV_VIEW="default"
BIN="./build_alps/cvfem_hex8_ns_ssgmg"
OUT="$SCRATCH/egal_out_threads"
mkdir -p "$OUT"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

for per in 20000 5000 2000 1000 500; do
    echo "### SFEM_GMG_DOFS_PER_THREAD=${per}, 242500 dofs, 72 cores, 60 V-cycles ###"
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores \
                  SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 \
                  SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
                  SFEM_GMG_EGAL=1 SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=60 \
                  SFEM_GMG_DOFS_PER_THREAD=${per} SFEM_VERIFY_TOL=1e9 \
                  stdbuf -oL ${BIN} ${OUT}" 2>&1 \
        | grep -E "ndof|lin_it_total|t_solve|us_per_lin_it|smooth\[L0\]|smooth\[L1\]|op\[L0\]|top-level phases"
    echo
done
