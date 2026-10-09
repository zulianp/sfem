#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-coarse-op-rc
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=coarse_op_rc_%j.out
#SBATCH --error=coarse_op_rc_%j.err

# Is the coarse operator's velocity/pressure inconsistency the Rhie-Chow coefficient?
#
# SFEM_GMG_CHECK=1 compares the REDISCRETISED coarse operator A_c against the Galerkin
# composite R A_f P on a smooth coarse vector, and reports a best-fit scale per component.
# A rediscretised coarse operator is not meant to EQUAL R A P -- the two differ by a fixed
# factor from the h-scaling convention -- so what matters is whether every component carries
# the SAME factor. Measured serially on the cavity (N 4, level 2) they do not:
#
#     coarse-op 0 scale:  ux 0.8432  uy 0.7938  uz 0.7846  p 0.0979  -> p/v 0.1213
#     coarse-op 0 after-scale:  ux 0.4941  uy 0.2997  uz 0.3169  p 0.8043
#
# The three velocity components agree with each other to within 7%; the pressure row is out
# by a factor of eight. And `after-scale` -- the residual left once each component's own best
# fit is removed -- is 0.80 on the pressure row, so it is not a scalar multiple either.
#
# The driver already names the suspect. Rhie-Chow's coefficient is Df = rc_scale * h^2 / (2 mu),
# the only term that depends on the lattice spacing outright, so halving the resolution per
# level QUADRUPLES Df and the coarse pressure block ends up far stiffer than the fine block
# whose error it is supposed to correct. SFEM_GMG_RC_DECAY rescales it per level and DEFAULTS
# TO 1, i.e. no decay at all; the code notes 0.25 as the value that holds Df at the fine
# level's. If the stabilisation is the cause, 0.25 should pull the pressure scale toward the
# velocity scales and drop the after-scale residual. If it does not, the inconsistency is in
# the discretisation as a whole and no per-level rescaling will reach it.
#
# SERIAL, deliberately. The check_transfers sums are accumulated over LOCAL ranges with no
# reduction, so at more than one rank they double-count shared and ghost entries and are not
# global inner products at all -- the same run at two ranks reported adjoint ratios of 1.159
# and 0.625 where the identity demands 1. At one rank local sums ARE global, so this is the
# only scope in which the diagnostic currently means anything. Making it collective is a
# separate change; this experiment does not wait on it.
#
# Two hierarchy depths, because Df's error compounds per level: refine level 2 gives two
# levels and level 4 gives three, so a stabilisation-driven defect should worsen with depth
# while a discretisation-wide one should not.

set -u

# Locate the spike root -- the directory holding CMakeLists.txt, drivers/ and scripts/.
# BASH_SOURCE is NOT usable on its own: Slurm copies the batch script to
# /var/spool/slurmd/job<id>/slurm_script, so under sbatch the script's own path says nothing
# about where the tree is. SLURM_SUBMIT_DIR is the reliable one.
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
BIN="${CVFEM_BIN:-./build/cvfem_hex8_ns_ssgmg}"
OUT="${OUT:-$SCRATCH/cvfem_coarse_op_rc/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"

[ -x "$BIN" ] || { echo "error: no binary at $BIN (build with scripts/alps_env.sh first)" >&2; exit 1; }

export OMP_PROC_BIND=true
export OMP_PLACES=cores

# One run, logged in full and filtered for the lines this experiment is about. The raw log is
# written from the first line so a slow arm is distinguishable from a stuck one, which a bare
# `| grep` pipeline is not: it selects end-of-run fields and block-buffers on top of that.
run_arm() {
    local tag="$1"; shift
    echo "### ${tag} ###"
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}" 2>&1 \
        | tee "$RAW/${tag}.log" \
        | stdbuf -oL grep -aE "coarse-op|adjoint |nnodes:|ndof:|gmg: |newton_converged|lin_it_total|coarse LU"
    echo
}

COMMON="SFEM_CASE=cavity SFEM_N=4 SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_FGMRES=1 \
        SFEM_SMOOTHER=vanka SFEM_ENABLE_OUTPUT=0 OMP_PROC_BIND=true OMP_PLACES=cores"

for lvl in 2 4; do
    for decay in 1.0 0.5 0.25; do
        # Consistency: one Newton step, one linear iteration -- the diagnostic prints during
        # setup, so this measures the operator without paying for a solve.
        run_arm "diag_L${lvl}_rc${decay}" \
            "OMP_NUM_THREADS=72 ${COMMON} SFEM_ELEMENT_REFINE_LEVEL=${lvl} \
             SFEM_GMG_RC_DECAY=${decay} SFEM_GMG_CHECK=1 \
             SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

        # Convergence: does a change in the coarse operator's consistency actually buy
        # iterations? The consistency numbers are the diagnosis; this is the consequence.
        run_arm "conv_L${lvl}_rc${decay}" \
            "OMP_NUM_THREADS=72 ${COMMON} SFEM_ELEMENT_REFINE_LEVEL=${lvl} \
             SFEM_GMG_RC_DECAY=${decay}"
    done
done

echo "raw logs: $RAW"
