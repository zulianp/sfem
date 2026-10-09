#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-vanka-P3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_vanka_P3_%j.out
#SBATCH --error=nozzle_vanka_P3_%j.err
# A/B of the multiplicative Vanka sweep on a per-macro lattice stencil (build_p2) against the
# global-BSR row walk (build_p1). Both builds carry the OpenMP-BLAS FGMRES and the 1e-3 linear
# tolerance default; the tolerance is set explicitly anyway. Interleaved. Warped FDA nozzle,
# macro core 2, throat Re 500, FGMRES (SFEM_FGMRES=1, restart 480) + geometric multigrid.
# Complete logs via scripts/cvrun.sh; the phase breakdown and smesh trace land per run.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_vanka/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> <build> <seconds> <level> [extra env...]
    local tag=$1 build=$2 secs=$3 lvl=$4; shift 4; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it_total|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux|vanka_setup" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/$build/cvfem_hex8_ns_ssgmg" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 SFEM_LSOLVE_RTOL=1e-3 \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" "$@" )
    echo "--- $tag exit $?"
}
# build_p3 adds the threaded coarse dense-LU factorisation on top of build_p2, so p2 vs p1 is
# the Vanka stencil alone and p3 vs p2 is the factorisation alone.
run L4_p1_a build_p1 150 4
run L4_p2_a build_p2 150 4
run L4_p3_a build_p3 150 4
run L4_p1_b build_p1 150 4
run L4_p2_b build_p2 150 4
run L4_p3_b build_p3 150 4
run L8_p3   build_p3 420 8
run L8_p2   build_p2 420 8
run L8_p1   build_p1 420 8
echo "=== DONE $(date)"
