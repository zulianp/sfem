#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-vankasp-P12
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_vankasp_P12_%j.out
#SBATCH --error=nozzle_vankasp_P12_%j.err
# A/B of the Vanka smoother with single-precision storage (build_p8: per-cell factors and the
# row walk's matrix values as float, all arithmetic in double) against the committed double
# storage (build_p7, HEAD a0851e226). Driver defaults otherwise: linear tolerance 1e-3, Galerkin
# coarse operators, 2 sweeps. Interleaved. Warped FDA nozzle, macro core 2, throat Re 500,
# FGMRES (SFEM_FGMRES=1, restart 480) + geometric multigrid. Complete logs via
# scripts/cvrun.sh; phase breakdown per run.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_vankasp/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> <build> <seconds> <level>
    local tag=$1 build=$2 secs=$3 lvl=$4; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it_total|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/$build/cvfem_hex8_ns_ssgmg" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" )
    echo "--- $tag exit $?"
}
run L4_dp_a build_p7 90 4
run L4_sp_a build_p8 90 4
run L4_dp_b build_p7 90 4
run L4_sp_b build_p8 90 4
run L8_sp_a build_p8 240 8
run L8_dp_a build_p7 240 8
run L8_sp_b build_p8 240 8
run L8_dp_b build_p7 240 8
echo "=== DONE $(date)"
