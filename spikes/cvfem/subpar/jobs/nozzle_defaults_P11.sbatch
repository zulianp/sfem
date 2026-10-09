#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-defaults-P11
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_defaults_P11_%j.out
#SBATCH --error=nozzle_defaults_P11_%j.err
# The warped FDA nozzle at throat Re 500 on the driver's DEFAULTS alone (build_p7): linear
# tolerance, coarse operators and sweeps are not set, so this is what a run that names only the
# case and the multigrid now gets. FGMRES (SFEM_FGMRES=1, restart 480) + geometric multigrid.
# Complete logs via scripts/cvrun.sh; phase breakdown per run. XDMF-ready output kept.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_defaults/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> <seconds> <level>
    local tag=$1 secs=$2 lvl=$3; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it_total|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/build_p7/cvfem_hex8_ns_ssgmg" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" )
    echo "--- $tag exit $?"
}
run L4_defaults 120 4
run L8_defaults 300 8
echo "=== DONE $(date)"
