#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-egal-1arm
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=egal_converge_1arm_%j.out
#SBATCH --error=egal_converge_1arm_%j.err

# Full Newton solve to convergence, probe against element-wise Galerkin.
#
# Every convergence figure so far came either from a marginal regime where both
# constructions stagnated, or from a fixed-work NL_MAX_IT=2 comparison. The fixed-work
# run is what makes the cost numbers trustworthy -- identical iteration counts -- but it
# stops before convergence, so it cannot show the solver reaches a solution at all with
# the new coarse operators. This does.

set -u
# Locate the spike root -- the directory holding CMakeLists.txt, drivers/ and scripts/.
# Resolved rather than hardcoded, so the job does not carry one user's $SCRATCH layout.
#
# BASH_SOURCE is NOT usable on its own here. Slurm copies the batch script to
# /var/spool/slurmd/job<id>/slurm_script before running it, so under sbatch the script's
# own path says nothing about where the tree is -- it resolves to /var/spool/slurmd and
# every relative path after it fails. SLURM_SUBMIT_DIR, the directory sbatch was invoked
# from, is the reliable one; the rest are for running the script directly.
_cvfem_probe=scripts/alps_env.sh
SPIKE_ROOT=""
for _cand in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." "${CVFEM_DIR:-}" \
             "$(dirname "${BASH_SOURCE[0]}")/.."; do
    [ -n "${_cand}" ] && [ -f "${_cand}/${_cvfem_probe}" ] || continue
    SPIKE_ROOT="$(cd "${_cand}" && pwd)"; break
done
[ -n "${SPIKE_ROOT}" ] || {
    echo "error: could not locate spikes/cvfem; sbatch from there or set CVFEM_DIR" >&2; exit 1; }
cd "$SPIKE_ROOT"

UENV_IMAGE="prgenv-gnu/24.11:v2"
UENV_VIEW="default"
BIN="./build_alps/cvfem_hex8_ns_ssgmg"
OUT="$SCRATCH/egal_out"
mkdir -p "$OUT"

export OMP_PROC_BIND=true
export OMP_PLACES=cores

for eg in 1; do
    [ "$eg" = 0 ] && name="probe" || name="element-wise"
    echo "### ${name} : full Newton, 108 macro-elements, L=4, 72 threads ###"
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                  SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 \
                  SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
                  SFEM_GMG_EGAL=${eg} ${BIN} ${OUT}" \
        2>&1 | grep -E "nnodes|ndof|newton_converged|t_solve|galerkin_assembly|u_linf:|lin_it_total"
    echo
done
