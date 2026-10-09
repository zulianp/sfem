#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-bench-many
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=bench_ab_many_%j.out
#SBATCH --error=bench_ab_many_%j.err
# cvfem_sshex8_bench for several builds in one allocation, interleaved rep by rep so drift
# between them cannot masquerade as a difference.
#
#   sbatch --export=ALL,BENCHES="ref=/path/a new=/path/b" jobs/bench_ab_many.sbatch
set -u
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep "^uenv start" ~/init.sh | sed "s/^uenv start //")
for pair in "4:16" "8:8"; do L=${pair%%:*}; M=${pair##*:}
  for rep in 1 2 3; do
    for nb in $BENCHES; do name=${nb%%=*}; bin=${nb#*=}
      echo "### L=$L macros=$M rep=$rep $name"
      uenv run $UENV_ARGS -- env SFEM_BENCH_REPS=5 SFEM_BENCH_LEVELS=$L SFEM_BENCH_MACROS=$M \
          timeout 240 stdbuf -oL "$bin" 2>&1 | grep -E "^[0-9]+ +[0-9]" | tail -1
    done
  done
done
echo "=== DONE $(date) ==="
