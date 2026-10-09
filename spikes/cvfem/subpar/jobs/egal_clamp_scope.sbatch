#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-clamp-scope
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=egal_clamp_%j.out
#SBATCH --error=egal_clamp_%j.err

# Should the clamp apply only to the coarse space?
#
# The clamp changes the team size, and with schedule(static) a different team size is a
# different partition of the same array. The transfers are not clamped, so a coarse vector is
# written across 72 threads by restriction and read across 30 by the coarse smoother; a
# V-cycle crosses that boundary at every level, with per-call omp_set_num_threads churn on
# top. So there are two effects pulling opposite ways -- less work per thread on a small
# level, against a partition that no longer matches the one the transfers used.
#
#   A  per=1000, no floor      L0=72 L1=30 L2=3  L3=1   (current default)
#   B  no clamp at all         L0=72 L1=72 L2=72 L3=72
#   C  clamp below 10000 dofs  L0=72 L1=72 L2=3  L3=1   (coarse space only)
#   D  clamp below 50000 dofs  L0=72 L1=30 L2=3  L3=1   (same as A here, a control)
#
# B is the case the clamp was originally added to prevent -- an 81-node level cost 156 us on
# one thread and 14.2 ms on eight -- so if B is not catastrophic that finding needs revisiting
# too. Three repeats, fixed work, identical iteration counts.

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
OUT="$SCRATCH/egal_out_clamp"
mkdir -p "$OUT"

run() {  # label per below
    echo "### $1 : DOFS_PER_THREAD=$2 CLAMP_BELOW=$3, 242500 dofs, 72 cores, 60 V-cycles ###"
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                  SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 \
                  SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
                  SFEM_GMG_EGAL=1 SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=60 \
                  SFEM_GMG_DOFS_PER_THREAD=$2 SFEM_GMG_CLAMP_BELOW=$3 \
                  SFEM_VERIFY_TOL=1e9 stdbuf -oL ${BIN} ${OUT}" 2>&1 \
        | grep -E "top-level phases|smooth\[L0\]|smooth\[L1\]|smooth\[L2\]|restrict|prolong|coarse_solve|t_solve"
    echo
}

for rep in 1 2 3; do
    run "rep${rep} A current"       1000 0
    run "rep${rep} B no-clamp"         1 0
    run "rep${rep} C coarse-only"   1000 10000
done
