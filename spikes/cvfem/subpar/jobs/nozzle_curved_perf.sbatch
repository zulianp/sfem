#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-curved-perf
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_curved_perf_%j.out
#SBATCH --error=nozzle_curved_perf_%j.err
# The semi-structured kernels after the curved-macro-element branch, against the committed
# kernels, on affine boxes -- where the branch must cost nothing. Alternating, one allocation.
set -u
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep "^uenv start" ~/init.sh | sed "s/^uenv start //")
NEW=${NEW_BENCH:-$SLURM_SUBMIT_DIR/build_nozzle/cvfem_sshex8_bench}
# REF_BENCH should be built from the same tree with only the change under test removed:
# comparing against an older tree measures everything else that changed in between.
REF=${REF_BENCH:-$SCRATCH/cvfem_nozref/build_ref/cvfem_sshex8_bench}
for pair in "4:16" "8:8"; do L=${pair%%:*}; M=${pair##*:}
  for rep in 1 2 3; do for which in ref new; do
    [ $which = ref ] && BIN=$REF || BIN=$NEW
    echo "### L=$L macros=$M rep=$rep $which"
    uenv run $UENV_ARGS -- env SFEM_BENCH_REPS=5 SFEM_BENCH_LEVELS=$L SFEM_BENCH_MACROS=$M timeout 240 stdbuf -oL "$BIN" 2>&1 | grep -E "^(L |[0-9]+ +[0-9])" | tail -2
  done; done
done
echo "new $NEW"
echo "ref $REF"
echo "=== DONE $(date) ==="
