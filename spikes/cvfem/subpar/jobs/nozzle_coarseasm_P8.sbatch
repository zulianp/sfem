#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-coarseasm-P8
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_coarseasm_P8_%j.out
#SBATCH --error=nozzle_coarseasm_P8_%j.err
# The coarsest dense matrix assembled with hessian_bsr instead of probed (build_p5) against the
# probe (build_p4, LAPACK getrf in both). First a short gated run (SFEM_GMG_CHECK=1: assembled vs
# probed coarse LU on the nozzle's own coarsest level), then the full Re 500 A/B at 116,212 and
# 893,924 dof, interleaved. Warped FDA nozzle, macro core 2, FGMRES (SFEM_FGMRES=1, restart 480)
# + geometric multigrid, linear tolerance 1e-3. Complete logs via scripts/cvrun.sh.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_coarseasm/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
run() {  # run <tag> <build> <seconds> <level> [extra env...]
    local tag=$1 build=$2 secs=$3 lvl=$4; shift 4; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it_total|t_solve|ndof:|gmg:|warped|line search|retrying|inflow flux|coarse LU" \
      uenv run $UENV_ARGS -- timeout "$secs" zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/$build/cvfem_hex8_ns_ssgmg" \
        SFEM_CASE=nozzle SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 SFEM_LSOLVE_RTOL=1e-3 \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" "$@" )
    echo "--- $tag exit $?"
}
run gate_L4   build_p5 240 4 SFEM_NOZZLE_RE=1 SFEM_GMG_CHECK=1
run L4_p4_a   build_p4 150 4 SFEM_NOZZLE_RE=500
run L4_p5_a   build_p5 150 4 SFEM_NOZZLE_RE=500
run L4_p4_b   build_p4 150 4 SFEM_NOZZLE_RE=500
run L4_p5_b   build_p5 150 4 SFEM_NOZZLE_RE=500
run L8_p5     build_p5 420 8 SFEM_NOZZLE_RE=500
run L8_p4     build_p4 420 8 SFEM_NOZZLE_RE=500
echo "=== DONE $(date)"
