#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-lrtol-R1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_lrtol_R1_%j.out
#SBATCH --error=nozzle_lrtol_R1_%j.err
# The warped FDA nozzle at throat Re 500 (macro core 2), FGMRES (SFEM_FGMRES=1, restart 480) +
# geometric multigrid, linear relative tolerance inside Newton: 1e-4 and 1e-3 against the
# default 1e-8 (measured in W1/W2: L4 2,386 its, L8 4,020 its with the first solve capped at
# 1000). Same binary as W1/W2. Complete logs via scripts/cvrun.sh.
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
run c2L8_mg_r1e-3 540 8 1 1e-3
run c2L8_mg_r1e-4 600 8 1 1e-4
run c2L4_mg_r1e-3 120 4 1 1e-3
run c2L4_mg_r1e-4 120 4 1 1e-4
echo "=== DONE $(date)"
