#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-lapack-P4
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_lapack_P4_%j.out
#SBATCH --error=nozzle_lapack_P4_%j.err
# A/B of the coarse dense LU through LAPACK getrf (build_p4) against the hand-written serial
# loop (build_p1). Both carry the OpenMP-BLAS FGMRES, the 1e-3 linear tolerance default and the
# same Vanka sweep, so the difference is the factorisation and its triangular solves alone.
# Interleaved. Warped FDA nozzle, macro core 2, throat Re 500, FGMRES (SFEM_FGMRES=1, restart
# 480) + geometric multigrid. Complete logs via scripts/cvrun.sh; phase breakdown per run.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_lapack/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> <build> <seconds> <level> [extra env...]
    local tag=$1 build=$2 secs=$3 lvl=$4; shift 4; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it_total|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux|coarse LU" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/$build/cvfem_hex8_ns_ssgmg" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 SFEM_LSOLVE_RTOL=1e-3 \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" "$@" )
    echo "--- $tag exit $?"
}
run L4_p1_a build_p1 150 4
run L4_p4_a build_p4 150 4
run L4_p1_b build_p1 150 4
run L4_p4_b build_p4 150 4
run L8_p4   build_p4 420 8
run L8_p1   build_p1 420 8
echo "=== DONE $(date)"
