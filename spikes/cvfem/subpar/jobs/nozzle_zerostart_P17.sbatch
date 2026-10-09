#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-zerostart-P17
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_zerostart_P17_%j.out
#SBATCH --error=nozzle_zerostart_P17_%j.err
# A/B of zero-start pre-smoothing: Multigrid tells its smoothers when their solution is zero
# (coarse levels always, the finest on the first cycle when the caller promises a zero x), and
# StationaryIteration then takes b as its first residual instead of applying the operator to
# zero. build_p12 links SFEM from installations/sfem-omp-zsg (HEAD plus those two algebra edits)
# and carries the driver's set_initial_guess_zero(true); build_p11 is the same tree without it
# (HEAD 0da852426 plus the Galerkin pattern cache, on the shared sfem-omp), so the difference is
# the zero start alone. Driver defaults otherwise. Interleaved. Warped FDA nozzle, macro core 2,
# throat Re 500, FGMRES (SFEM_FGMRES=1, restart 480) + geometric multigrid.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_zerostart/$SLURM_JOB_ID; mkdir -p "$OUT"
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
run L4_old_a build_p11 90 4
run L4_new_a build_p12 90 4
run L4_old_b build_p11 90 4
run L4_new_b build_p12 90 4
run L8_new_a build_p12 180 8
run L8_old_a build_p11 180 8
run L8_new_b build_p12 180 8
run L8_old_b build_p11 180 8
echo "=== DONE $(date)"
