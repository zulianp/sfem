#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-egal-re1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=egal_re1_%j.out
#SBATCH --error=egal_re1_%j.err

# Solver robustness against Reynolds number, element-wise Galerkin throughout.
#
# One Reynolds number per job, taken from the RE environment variable.
#
# A single Re run is 13 minutes at Re=100 and so belongs on debug; only the six-point sweep
# as one job exceeded 30 minutes. Splitting it also gets results far sooner -- normal was
# 1793 jobs deep and would not have started for four hours, against 20 on debug. The debug
# QOS allows 2 submitted and 1 running, so these go two at a time.
#
# Output is unfiltered and line-buffered on purpose. Piping through grep block-buffers, so
# nothing reaches the .out file until the srun exits -- a run that is still going, or that
# dies on the wall clock, then yields no diagnostic at all. The full log carries the
# per-Newton residual trail, which is exactly what distinguishes "diverging" from
# "converging slowly" while the job is still running. Filter when reading, not when writing.
#
# Order matters: the hardest case runs first. If the highest Re converges the range below
# it is not in question, and if it fails the limit can be bisected. Sweeping upward would
# spend the queue on the cases least likely to tell us anything.
#
# The run must go to completion, not be capped: the driver does Reynolds continuation
# in exactly two stages -- solve at Re=1 with rho = mu/(U Ly), then jump straight to the
# physical density -- so a capped run measures the Re=1 stage and says nothing about the
# target. That is how an earlier scoping attempt here produced six runs that all solved
# the same problem. Whether the single jump from Re=1 to the physical Re survives is
# itself the robustness question.
#
# Poiseuille keeps the exact solution parabolic at every Re, since the convective term
# vanishes at the solution, so u_linf remains a real correctness check while the
# linearised systems become convection-dominated.
#
# Re = rho * U * Ly / mu with rho = U = Ly = 1, so mu = 1/Re.

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

export OMP_PROC_BIND=close
export OMP_PLACES=cores

re="${RE}"
mu=$(python3 -c "print(1.0/${re})")
OUT="$SCRATCH/egal_out_re${re}"
mkdir -p "$OUT"
echo "### Re=${re} (mu=${mu}), element-wise, 108 macro-elements, L=4, 72 threads ###"
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
    bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores \
              SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 \
              SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
              SFEM_GMG_EGAL=1 SFEM_MU=${mu} SFEM_VERIFY_TOL=1e9 \
              stdbuf -oL -eL ${BIN} ${OUT}" 2>&1
