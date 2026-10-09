#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-smooth-P9
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_smooth_P9_%j.out
#SBATCH --error=nozzle_smooth_P9_%j.err
# Vanka sweeps per level (pre and post), 1 / 2 / 3, with element-wise Galerkin coarse operators
# (SFEM_GMG_GALERKIN=2, which halved the linear iterations against rediscretised ones). Earlier: the
# fine-level smoother is ~70% of the 893,924-dof run, so fewer sweeps may win even with more
# iterations. build_p4 (OpenMP-BLAS FGMRES, 1e-3 tolerance, LAPACK coarse LU); the coarse
# factorisation is per Newton step and does not change with the sweep count.
# Warped FDA nozzle, macro core 2, throat Re 500, FGMRES (SFEM_FGMRES=1, restart 480) +
# geometric multigrid. Complete logs via scripts/cvrun.sh; phase breakdown per run.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_smooth_gal/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> <seconds> <level> <sweeps>
    local tag=$1 secs=$2 lvl=$3 sw=$4; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it_total|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/build_p4/cvfem_hex8_ns_ssgmg" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 SFEM_LSOLVE_RTOL=1e-3 SFEM_GMG_SMOOTH=$sw SFEM_GMG_GALERKIN=2 \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" )
    echo "--- $tag exit $?"
}
run L4_s2 150 4 2
run L4_s1 200 4 1
run L4_s3 150 4 3
run L8_s2 480 8 2
run L8_s1 540 8 1
run L8_s3 420 8 3
echo "=== DONE $(date)"
