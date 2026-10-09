#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-perf-P13
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_perf_P13_%j.out
#SBATCH --error=nozzle_perf_P13_%j.err
# Profile of the production set-up at HEAD f201fd76d (driver defaults: tolerance 1e-3, Galerkin coarse
# operators, 2 Vanka sweeps, single-precision Vanka storage), focused on the Vanka setup phase: 17.8 s of
# an 81 s run at 893,924 dof, of which the trace accounts for ~7.7 s. First-order upwind, FGMRES
# (SFEM_FGMRES=1, restart 480) + geometric multigrid, warped FDA nozzle Re 500. perf adds what the
# trace does not cover: pattern and scatter builds, allocation, page faults, memset/memcpy.
# Complete logs via scripts/cvrun.sh; perf follows the driver through the wrapper.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
DRIVER=$SPIKE_ROOT/build_p8/cvfem_hex8_ns_ssgmg
OUT=$SCRATCH/cvfem_nozzle_perf13/$SLURM_JOB_ID; mkdir -p "$OUT"
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

run c2L8 300 8 199
echo "=== DONE $(date)"
