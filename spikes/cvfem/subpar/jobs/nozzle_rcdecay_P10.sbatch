#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-rcdecay-P10
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_rcdecay_P10_%j.out
#SBATCH --error=nozzle_rcdecay_P10_%j.err
# Coarse operators, the fair comparison: rediscretised with the Rhie-Chow decay docs/MG_REDESIGN.md
# calls a requirement (SFEM_GMG_GALERKIN=0 SFEM_GMG_RC_DECAY=0.25) against element-wise Galerkin
# (SFEM_GMG_GALERKIN=2), which that document records as inconsistent with the default exact
# Rhie-Chow Jacobian. Both set explicitly, so the binary's default does not matter. build_p4:
# OpenMP-BLAS FGMRES, 1e-3 linear tolerance, LAPACK coarse LU. Warped FDA nozzle, macro core 2,
# throat Re 500, FGMRES (SFEM_FGMRES=1, restart 480) + geometric multigrid. Interleaved.
# Complete logs via scripts/cvrun.sh; phase breakdown per run.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_rcdecay/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> <seconds> <level> [extra env...]
    local tag=$1 secs=$2 lvl=$3; shift 3; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it_total|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/build_p4/cvfem_hex8_ns_ssgmg" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 SFEM_LSOLVE_RTOL=1e-3 \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" "$@" )
    echo "--- $tag exit $?"
}
run L4_g0d25   150 4 SFEM_GMG_GALERKIN=0 SFEM_GMG_RC_DECAY=0.25
run L4_g2      150 4 SFEM_GMG_GALERKIN=2
run L4_g0d50   150 4 SFEM_GMG_GALERKIN=0 SFEM_GMG_RC_DECAY=0.5
run L8_g0d25   420 8 SFEM_GMG_GALERKIN=0 SFEM_GMG_RC_DECAY=0.25
run L8_g2      300 8 SFEM_GMG_GALERKIN=2
echo "=== DONE $(date)"
