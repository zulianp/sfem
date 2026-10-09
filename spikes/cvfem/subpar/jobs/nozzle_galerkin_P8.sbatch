#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-galerkin-P8
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_galerkin_P8_%j.out
#SBATCH --error=nozzle_galerkin_P8_%j.err
# Coarse operators: rediscretised (SFEM_GMG_GALERKIN=0, the default -- the coarsest level's dense
# matrix is recovered by probing, ~560 ms per Newton step at 2,320 dof) against element-wise
# Galerkin assembled once per Newton step (SFEM_GMG_GALERKIN=2 -- no probing, a different coarse
# operator). build_p4: OpenMP-BLAS FGMRES, 1e-3 linear tolerance, LAPACK coarse LU.
# Warped FDA nozzle, macro core 2, throat Re 500, FGMRES (SFEM_FGMRES=1, restart 480) +
# geometric multigrid. Interleaved. Complete logs via scripts/cvrun.sh; phase breakdown per run.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_galerkin/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> <seconds> <level> <galerkin mode>
    local tag=$1 secs=$2 lvl=$3 gm=$4; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it_total|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux|coarse LU" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/build_p4/cvfem_hex8_ns_ssgmg" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 SFEM_LSOLVE_RTOL=1e-3 SFEM_GMG_GALERKIN=$gm \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" )
    echo "--- $tag exit $?"
}
run L4_g0_a 150 4 0
run L4_g2_a 150 4 2
run L4_g0_b 150 4 0
run L4_g2_b 150 4 2
run L8_g2   480 8 2
run L8_g0   420 8 0
echo "=== DONE $(date)"
