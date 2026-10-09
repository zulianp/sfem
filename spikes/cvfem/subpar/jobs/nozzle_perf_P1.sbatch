#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-perf-P1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:45:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_perf_P1_%j.out
#SBATCH --error=nozzle_perf_P1_%j.err
# Profile of the current production set-up: warped FDA nozzle, macro core 2, throat Re 500,
# first-order upwind, FGMRES (SFEM_FGMRES=1, restart 480) + geometric multigrid with
# multiplicative Vanka, linear rtol 1e-8 (the default). Same binary and settings as W1/W2, so
# the SFEM trace and phase breakdown of those runs apply; perf adds what they do not cover
# (the Krylov method, allocation, page faults, libc) and the hardware counters.
# Complete logs via scripts/cvrun.sh; perf follows the driver through the wrapper.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
DRIVER=$SPIKE_ROOT/build_w/cvfem_hex8_ns_ssgmg
OUT=$SCRATCH/cvfem_nozzle_perf/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) driver $DRIVER out $OUT $(date)"
run() {  # run <tag> <seconds> <level> <freq>
    local tag=$1 secs=$2 lvl=$3 freq=$4; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
      CVRUN_FILTER="^newton |^stage |highest Re SOLVED|newton_converged|lin_it|t_solve|ndof:|gmg:|warped|line search|retrying" \
      uenv run $UENV_ARGS -- timeout "$secs" \
        perf stat -e task-clock,page-faults,cycles:u,instructions:u,cache-misses:u,stalled-cycles-backend:u -o "$OUT/$tag.perfstat" -- \
        perf record -F "$freq" -o "$OUT/$tag.perf.data" -- \
        zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$DRIVER" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine \
        SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" )
    echo "--- $tag exit $?"
    uenv run $UENV_ARGS -- perf report -i "$OUT/$tag.perf.data" --no-children --sort dso,symbol \
        --percent-limit 0.3 --stdio 2>/dev/null | grep -v '^#' | grep -v '^$' > "$OUT/$tag.perf.txt"
    head -45 "$OUT/$tag.perf.txt"; cat "$OUT/$tag.perfstat"
    for sym in cvfem_ss::vanka_apply_mult sfem::FGMRES; do
        uenv run $UENV_ARGS -- perf annotate -i "$OUT/$tag.perf.data" --stdio -s "$(grep -o "[^ ]*${sym##*::}[^ ]*" "$OUT/$tag.perf.txt" | head -1)" \
            2>/dev/null | sort -t: -k1 -gr | head -40 > "$OUT/$tag.annotate_${sym##*::}.txt"
    done
}
run c2L4 300 4 499
run c2L8 1200 8 49
echo "=== DONE $(date)"
