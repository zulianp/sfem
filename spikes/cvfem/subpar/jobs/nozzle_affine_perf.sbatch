#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-nozzle-perf
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_affine_perf_%j.out
#SBATCH --error=nozzle_affine_perf_%j.err

# Two things jobs/nozzle_affine.sbatch leaves out.
#
# 1. The semi-structured apply after sscvfem_hoisted_cell, against the same code before it.
#    The change adds one centre Jacobian per MACRO element and nothing per micro element, so
#    it should be free; that is a claim to be measured, not asserted. REF_BENCH is built from
#    a copy of this tree whose src/ss/cvfem_sshex8_ns.hpp is the committed one. Both binaries
#    run in this one allocation, alternating, at matched problem size and two levels.
#
# 2. Flat core 4 with SFEM_GEOM=affine, the partner of the isoparametric core 4 measured in
#    job 4656808. Last, because its probed LU takes about 13 minutes.
set -u

SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$SPIKE_ROOT/CMakeLists.txt" ] || { echo "not the spike root: $SPIKE_ROOT" >&2; exit 1; }
cd "$SPIKE_ROOT"

export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
NEW_BENCH=${NEW_BENCH:-$SPIKE_ROOT/build_nozzle/cvfem_sshex8_bench}
REF_BENCH=${REF_BENCH:-$SCRATCH/cvfem_nozref/build_ref/cvfem_sshex8_bench}
DRIVER=$SPIKE_ROOT/build_nozzle/cvfem_hex8_ns_ssgmg
OUT=${OUT:-$SCRATCH/cvfem_nozzle/$SLURM_JOB_ID}
mkdir -p "$OUT"
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "new $NEW_BENCH"
echo "ref $REF_BENCH"

# Header line once, then the result row of each run. The columns are the bench's own; the
# naive-vs-macro agreement columns must stay at round-off in both.
for pair in "4:16" "8:8"; do
    L=${pair%%:*}; M=${pair##*:}
    for rep in 1 2 3; do
        for which in ref new; do
            [ $which = ref ] && BIN=$REF_BENCH || BIN=$NEW_BENCH
            echo "### L=$L macros=$M rep=$rep $which"
            uenv run $UENV_ARGS -- env SFEM_BENCH_REPS=5 SFEM_BENCH_LEVELS=$L SFEM_BENCH_MACROS=$M \
                timeout 240 stdbuf -oL "$BIN" > "$OUT/bench_L${L}_${which}_${rep}.log" 2>&1
            grep -E "^(L |[0-9]+ +[0-9])" "$OUT/bench_L${L}_${which}_${rep}.log" | tail -2
        done
    done
done

uenv run $UENV_ARGS -- env CVRUN_DIR="$OUT" CVRUN_OUT="$OUT/flat_affine_core4" \
    CVRUN_FILTER='ndof:|^nozzle: (station|sum|inflow)|newton_converged|t_precond|t_solve' \
    timeout 1000 bash scripts/cvrun.sh flat_affine_core4 "$DRIVER" \
    SFEM_CASE=nozzle SFEM_NOZZLE_RE=1 SFEM_FGMRES=1 SFEM_GEOM=affine SFEM_ELEMENT_REFINE_LEVEL=1 \
    SFEM_GMG=0 SFEM_PRECOND=direct SFEM_DIRECT_MAX_DOF=40000 \
    SFEM_NOZZLE_NCORE=4 SFEM_NOZZLE_NBORE=2 SFEM_NOZZLE_NOUTER=2 "SFEM_NOZZLE_NAXIAL=8 6 16 24" || true

echo "=== DONE $(date) ==="
