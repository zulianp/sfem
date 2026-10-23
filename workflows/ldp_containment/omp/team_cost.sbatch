#!/bin/bash
#SBATCH --job-name=ldp-omp-team
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# A QP iteration's synchronisation, a region per pass against one persistent region (team_cost.c),
# into $OUT/team_cost.csv: ranks, threads, n (doubles per rank: FF200's 150,525 dofs or FF200
# refined's 973,812 over the ranks), us per iteration of each variant, and the checksum's largest
# relative difference from the region per pass.
# Build on the login node first, under the same uenv:
#   mpicc -O2 -fopenmp team_cost.c -o team_cost -lm
#   OUT=... sbatch team_cost.sbatch
set -eu
: "${OUT:?}"
HERE=${SLURM_SUBMIT_DIR:-$(pwd)}
mkdir -p "$OUT"
export OMP_PROC_BIND=true OMP_PLACES=cores MPICH_GPU_SUPPORT_ENABLED=0
CSV=$OUT/team_cost.csv
echo "ranks,threads,n,regions_us,fused_us,team_us,diss_us,work_us,chunk256_us,chunk1024_us,chunk4096_us,max_rel_diff" > "$CSV"
for shape in "1 72 150525" "1 36 150525" "1 18 150525" "4 18 37631" "4 36 37631" "4 72 37631" "16 18 9408" \
             "1 72 973812" "4 72 243453" "16 18 60864"; do
  set -- $shape
  OMP_NUM_THREADS=$2 srun -N 1 -n "$1" --cpus-per-task="$2" --cpu-bind=cores --distribution=block:block \
    "$HERE/team_cost" "$3" | tr ' ' ',' >> "$CSV"
done
cat "$CSV"
