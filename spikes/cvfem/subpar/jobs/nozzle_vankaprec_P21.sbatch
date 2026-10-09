#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-vprec-P21
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_vankaprec_P21_%j.out
#SBATCH --error=nozzle_vankaprec_P21_%j.err
# The Vanka storage precision as a setting (build_p14: SFEM_VANKA_PRECISION, single by default,
# double on request; computation in double either way), both settings on one binary, and the
# default against build_p12 (single precision hard-coded, same zero-start build) to show that
# making it a setting costs nothing. Driver defaults otherwise. Interleaved. Warped FDA nozzle,
# macro core 2, throat Re 500, FGMRES (SFEM_FGMRES=1, restart 480) + geometric multigrid.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_vprec/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> <build> <seconds> <level> [extra env...]
    local tag=$1 build=$2 secs=$3 lvl=$4; shift 4; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it_total|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/$build/cvfem_hex8_ns_ssgmg" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" "$@" )
    echo "--- $tag exit $?"
}
run L4_p12        build_p12 90 4
run L4_single     build_p14 90 4
run L4_double     build_p14 90 4 SFEM_VANKA_PRECISION=double
run L8_single     build_p14 150 8
run L8_p12        build_p12 150 8
run L8_double     build_p14 200 8 SFEM_VANKA_PRECISION=double
run L8_single_b   build_p14 150 8
echo "=== DONE $(date)"
