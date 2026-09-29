#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-coarse-op-cases
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=coarse_op_cases_%j.out
#SBATCH --error=coarse_op_cases_%j.err

# Does the Rhie-Chow rescaling that fixes coarse-operator CONSISTENCY ever buy ITERATIONS?
#
# The first sweep (coarse_op_rc_decay.sbatch, job 4714344) established the mechanism on the
# cavity. SFEM_GMG_RC_DECAY scales Df = rc_scale * h^2 / (2 mu) per level, and holding it
# fixed at 0.25 improved the coarse operator's pressure consistency about fivefold:
#
#     L2   decay 1.00 -> p rel 5.5316  p scale 0.0979  after-scale 0.8043  lin_it 61
#          decay 0.25 -> p rel 1.0725  p scale 0.4768  after-scale 0.5130  lin_it 61
#
# The velocity columns were bit-identical in every arm, which is the signature that Df
# touches the continuity row alone -- exactly as the code predicts. But lin_it_total did not
# move by a single iteration, and after-scale p was still 0.51 at the best arm, so the
# pressure row is still not a scalar multiple of R A P.
#
# The cavity is the wrong place to conclude from. It is closed and Dirichlet on the whole
# skin, so the correction at boundary nodes is zeroed after prolongation and a defect confined
# to those rows is masked exactly; the driver says so where it explains why the do-nothing
# outflow is the only configuration that exposes them. It is also 2,916 dof, far under any
# size where the coarse grid has to carry real work.
#
# So: the same three decays, on cases where the coarse correction is documented to struggle,
# at sizes that are not toy.
#
#   poiseuille L4 closed    33,124 dof   59 lin_it   the working reference
#   poiseuille L4 natural   33,124 dof   79 lin_it   SAME geometry, open outlet, +34%
#   poiseuille L8 closed   242,500 dof 2056 lin_it   saturating, 4 levels
#   step       N8 L2        47,268 dof  191 lin_it   L-shape AND open outlet
#   nozzle     c2 L2        15,740 dof 4458 lin_it   do-nothing outflow, hardest cycle here
#
# The poiseuille pair is the control the driver asks for: SFEM_OUTLET=natural differs from the
# working case in the outlet treatment and nothing else, so it separates outlet treatment from
# geometry, which the step confounds.
#
# lin_it_total AGGREGATES OVER CONTINUATION STAGES, and the cases do not share a stage count
# (nozzle 11, step 5, poiseuille 5). Compare arms WITHIN a case only; the absolute numbers
# above are not comparable to each other.
#
# SFEM_GMG_CHECK=4 is deliberately NOT used. Its cgc probe builds the coarse operator as
# literal R A P whenever SFEM_GMG_GALERKIN is non-zero, for which the surviving fraction is
# zero by construction -- it measured 0.0000 on velocity and nan on pressure, the pressure
# mode being annihilated by the gauge before it is prolonged. Measuring rediscretisation error
# with it needs SFEM_GMG_GALERKIN=0, which is a different configuration from the one this
# campaign runs, so it belongs to its own experiment rather than an arm here.

set -u

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
OUT="${OUT:-$SCRATCH/cvfem_coarse_op_cases/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

export OMP_PROC_BIND=true
export OMP_PLACES=cores

run_arm() {
    local tag="$1"; shift
    echo "### ${tag} ###"
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}" 2>&1 \
        | tee "$RAW/${tag}.log" \
        | stdbuf -oL grep -aE "coarse-op|nnodes:|ndof:|gmg: |newton_converged|lin_it_total"
    echo
}

GMG="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_FGMRES=1 SFEM_SMOOTHER=vanka SFEM_ENABLE_OUTPUT=0 \
     OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores"
POIS="SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3"
STEP="SFEM_CASE=step SFEM_U=1 SFEM_MU=0.02 SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=2"
NOZ="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_NOZZLE_NCORE=2 \
     SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 SFEM_NOZZLE_NAXIAL=\"4 3 8 12\" \
     SFEM_ELEMENT_REFINE_LEVEL=2 SFEM_FGMRES_RESTART=480"

# name : env  -- kept as parallel arrays so the loop below stays readable
CASE_NAMES="pois_closed_L4 pois_natural_L4 pois_closed_L8 step_N8L2 nozzle_c2L2"

case_env() {
    case "$1" in
        pois_closed_L4)  echo "$POIS SFEM_ELEMENT_REFINE_LEVEL=4" ;;
        pois_natural_L4) echo "$POIS SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_OUTLET=natural" ;;
        pois_closed_L8)  echo "$POIS SFEM_ELEMENT_REFINE_LEVEL=8" ;;
        step_N8L2)       echo "$STEP" ;;
        nozzle_c2L2)     echo "$NOZ" ;;
    esac
}

for cs in $CASE_NAMES; do
    CENV="$(case_env "$cs")"
    for decay in 1.0 0.5 0.25; do
        # Consistency: the diagnostic prints during setup, so one Newton step suffices and
        # the arm costs a fraction of a solve. These exit non-zero by design (NL_MAX_IT=1
        # reports "did not converge"); the script does not use set -e for that reason.
        run_arm "diag_${cs}_rc${decay}" \
            "${GMG} ${CENV} SFEM_GMG_RC_DECAY=${decay} SFEM_GMG_CHECK=1 \
             SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

        # Consequence: does better consistency actually buy iterations on this case?
        run_arm "conv_${cs}_rc${decay}" \
            "${GMG} ${CENV} SFEM_GMG_RC_DECAY=${decay}"
    done
done

echo "raw logs: $RAW"
