#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-fgmres-P2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_fgmres_P2_%j.out
#SBATCH --error=nozzle_fgmres_P2_%j.err
# A/B of FGMRES on SFEM's OpenMP BLAS (build_p1) against the serial FGMRES (build_w), both
# at SFEM_LSOLVE_RTOL=1e-3 set explicitly, interleaved. Warped FDA nozzle, macro core 2,
# throat Re 500, FGMRES (SFEM_FGMRES=1, restart 480) + geometric multigrid. Complete logs via
# scripts/cvrun.sh; the phase breakdown and smesh trace land per run.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_fgmres/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> <build> <seconds> <level> [extra env...]
    local tag=$1 build=$2 secs=$3 lvl=$4; shift 4; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it_total|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/$build/cvfem_hex8_ns_ssgmg" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 SFEM_LSOLVE_RTOL=1e-3 \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" "$@" )
    echo "--- $tag exit $?"
}
run L4_old_a build_w  150 4
run L4_new_a build_p1 150 4
run L4_old_b build_w  150 4
run L4_new_b build_p1 150 4
run L8_new   build_p1 480 8
run L8_old   build_w  480 8
echo "=== DONE $(date)"
