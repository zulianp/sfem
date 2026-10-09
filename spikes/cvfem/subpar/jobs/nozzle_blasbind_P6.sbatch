#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-blasbind-P6
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_blasbind_P6_%j.out
#SBATCH --error=nozzle_blasbind_P6_%j.err
# Why getrf on the 2,320-dof coarsest level takes ~580 ms, the cost of one core: the driver's
# OpenMP binds its master thread to one core (OMP_PROC_BIND=true), and the pthreads OpenBLAS
# may be creating its workers with that one-core mask. Measured on the coarse_factor phase of
# a short run (Re 1, one continuation stage) at 116,212 dof: binding on, binding off, and
# OpenBLAS forced to one thread. build_p4, FGMRES (SFEM_FGMRES=1) + geometric multigrid.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_blasbind/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> [extra env...]
    local tag=$1; shift; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |newton_converged|lin_it_total|t_solve|ndof:|coarse LU" \
      uenv run $UENV_ARGS -- timeout 240 zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/build_p4/cvfem_hex8_ns_ssgmg" \
        OMP_NUM_THREADS=72 OMP_PLACES=cores \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=1 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 SFEM_LSOLVE_RTOL=1e-3 \
        SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" "$@" )
    echo "--- $tag exit $?"
    grep -E "ndof:|newton_converged|coarse_factor|coarse_solve |t_solve|smooth\[L0\] " "$OUT/$tag.log"
}
run bind_true         OMP_PROC_BIND=true
run bind_false        OMP_PROC_BIND=false
run bind_true_ob1     OMP_PROC_BIND=true  OPENBLAS_NUM_THREADS=1
run bind_true_ob72    OMP_PROC_BIND=true  OPENBLAS_NUM_THREADS=72
run bind_spread       OMP_PROC_BIND=spread
echo "=== DONE $(date)"
