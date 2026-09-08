#!/bin/bash -l
#SBATCH --job-name=cvfem-vanka-re
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

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
BIN=$PWD/build_omp/cvfem_hex8_ns_ssgmg

# refuse to report numbers from a binary that predates the globalization
for sym in "line search failed" "highest Re SOLVED"; do
  if [ "$(strings $BIN | grep -c "$sym")" -eq 0 ]; then
    echo "ABORT: binary lacks '$sym' -- stale build"; exit 1; fi
done
echo "binary: $(date -r $BIN)  vanka syms: $(nm -C $BIN | grep -c make_diagonal_vanka)"

export OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=passive
export SFEM_CASE=poiseuille SFEM_NX=3 SFEM_NY=1 SFEM_NZ=1
export SFEM_ELEMENT_REFINE_LEVEL=16
export SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_GMG_EGAL=1
export SFEM_SMOOTHER=vanka SFEM_VERIFY_TOL=1e9 SFEM_RE_MAX_RETRY=16

for RE in 400 800 1600 3200 10000; do
  MU=$(python3 -c "print(1.0/$RE)")
  echo ""; echo "############ Re = $RE  (mu = $MU) ############"
  SFEM_MU=$MU timeout 300 stdbuf -oL $BIN /dev/null 2>&1 \
    | grep -E "ndof:|continuation:|retrying|abandoning|newton_converged|t_solve:|u_linf:|line search failed" \
    || echo "  [timed out at 300 s]"
done
echo ""; echo "ALL DONE"
