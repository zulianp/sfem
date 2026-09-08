#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-xfer-scale
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=egal_xfer_%j.out
#SBATCH --error=egal_xfer_%j.err

# Do the grid transfers scale with threads?
#
# After the thread-clamp fix the transfers are 33.4% of the cycle: restrict 20.9%, prolong
# 12.5%. Their absolute time did not move when everything else got 2.4x faster, which is the
# signature of something that was not sharing the bottleneck the clamp created.
#
# They are parallel -- sshex8_restrict and sshex8_prolongate carry OpenMP in their .impl.hpp,
# whatever the declaration headers suggest -- but they scatter with `#pragma omp atomic
# update`. If atomic contention is the cost, their time will be flat or worsen as threads are
# added, while the smoothers improve. That is the distinction this run makes, and it decides
# whether the fix is the two-pass scatter the spike already uses for its own kernels or
# something else entirely.

set -u
# Locate the spike root -- the directory holding CMakeLists.txt, drivers/ and src/.
# Resolved from this script's own path rather than a hardcoded scratch path, so the job
# does not carry one user's $SCRATCH layout and works from any checkout.
SPIKE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SPIKE_ROOT"

UENV_IMAGE="prgenv-gnu/24.11:v2"
UENV_VIEW="default"
BIN="./build_alps/cvfem_hex8_ns_ssgmg"
OUT="$SCRATCH/egal_out_xfer"
mkdir -p "$OUT"

for nt in 8 18 36 72; do
    echo "### OMP_NUM_THREADS=${nt}, 242500 dofs, 60 V-cycles ###"
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "OMP_NUM_THREADS=${nt} OMP_PROC_BIND=close OMP_PLACES=cores \
                  SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 \
                  SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
                  SFEM_GMG_EGAL=1 SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=60 \
                  SFEM_VERIFY_TOL=1e9 stdbuf -oL ${BIN} ${OUT}" 2>&1 \
        | grep -E "top-level phases|smooth\[L0\]|restrict|prolong|t_solve"
    echo
done
