#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-lrtol-R2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_lrtol_R2_%j.out
#SBATCH --error=nozzle_lrtol_R2_%j.err
# Companion to R1: the same linear-tolerance A/B without multigrid (FGMRES + Vanka alone) at
# L4, and with multigrid at L2 for the level trend. FGMRES (SFEM_FGMRES=1, restart 480) always.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
DRIVER=$SPIKE_ROOT/build_w/cvfem_hex8_ns_ssgmg
OUT=$SCRATCH/cvfem_nozzle_lrtol/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) driver $DRIVER out $OUT $(date)"
run() {  # run <tag> <seconds> <level> <gmg> <rtol>
    local tag=$1 secs=$2 lvl=$3 gmg=$4 rtol=$5; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux|station x|residual abs" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$DRIVER" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=$gmg SFEM_LSOLVE_RTOL=$rtol \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" )
    echo "--- $tag exit $?"
}
run c2L2_mg_r1e-8   150 2 1 1e-8
run c2L2_mg_r1e-3   150 2 1 1e-3
run c2L4_nomg_r1e-8 660 4 0 1e-8
run c2L4_nomg_r1e-3 660 4 0 1e-3
echo "=== DONE $(date)"
