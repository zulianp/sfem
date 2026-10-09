#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-gmg-consist-P22
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=gmg_consistency_P22_%j.out
#SBATCH --error=gmg_consistency_P22_%j.err
# Consistency of the operators with the coarse space, on the current build (build_p14: LAPACK
# coarse LU, Galerkin pattern cache, zero-start pre-smoothing, Vanka precision option).
#
# SFEM_GMG_CHECK=1 runs every gate the driver has: element locality of the fine Jacobian, the
# element-wise Galerkin identity at q = 1, element-wise Galerkin against P^T A P level by level,
# the constrained level against the probed composite (and its block diagonal), assembled vs
# probed coarse LU, structured vs matrix-free P, P reproducing constants, R = P^T, block split.
# The per-level gates fire at every Newton step, so capping Newton at 3 also exercises the
# pattern cache and a non-zero state.
#
# Two arms per case. SFEM_RC_EXACT_JAC=0 makes the matrix-free Jacobian the frozen-pg one the
# assembly builds, so every gate must pass to round-off -- that is the consistency statement.
# The default (1) differentiates through the nodal pressure gradient, which is not element-local,
# and the gap it leaves in the gates is the intended exact-vs-frozen difference, measured.
# Boundary conditions covered: closed Dirichlet box, do-nothing outlet, pressure port,
# manufactured solution (zero-mean gauge), warped FDA nozzle. FGMRES (SFEM_FGMRES=1) + GMG.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
BUILD=${BUILD:-build_p14}
OUT=$SCRATCH/cvfem_gmg_consist/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) build $BUILD out $OUT $(date)"
GATES="ndof:|gmg:|element-locality|egal |structured P|reproduces constants|adjoint|sum vs full|constraints level|coarse LU|coarse-op|rap level|newton [0-9]|MISMATCH|FAILED|BROKEN|NON-LOCAL"
run() {  # run <tag> <seconds> [env...]
    local tag=$1 secs=$2; shift 2; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out CVRUN_FILTER="$GATES" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/$BUILD/cvfem_hex8_ns_ssgmg" \
        SFEM_FGMRES=1 SFEM_GMG=1 SFEM_GMG_CHECK=1 SFEM_NL_MAX_IT=3 "$@" )
    echo "--- $tag exit $?"
}
CH="SFEM_LX=4 SFEM_LY=1 SFEM_LZ=1 SFEM_MU=0.01 SFEM_U=1 SFEM_CASE=poiseuille SFEM_BOUNDARY_MASK=1"
NOZ="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"
for ej in 0 1; do
    run box_ej$ej     150 $CH SFEM_N=2 SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_OUTLET=dirichlet SFEM_RC_EXACT_JAC=$ej
    run natural_ej$ej 120 $CH SFEM_N=3 SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_OUTLET=natural SFEM_RC_EXACT_JAC=$ej
    run port_ej$ej    120 $CH SFEM_N=3 SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_OUTLET=dirichlet \
        SFEM_PRESSURE_SIDESET=outlet SFEM_PRESSURE=0.16 SFEM_RC_EXACT_JAC=$ej
    run mms_ej$ej     120 SFEM_CASE=mms SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_MU=0.01 SFEM_RHO=1 SFEM_RC_EXACT_JAC=$ej
    run nozzle_ej$ej  200 $NOZ "SFEM_NOZZLE_NAXIAL=4 3 8 12" SFEM_RC_EXACT_JAC=$ej
done
echo "=== DONE $(date)"
