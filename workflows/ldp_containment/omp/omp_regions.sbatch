#!/bin/bash
#SBATCH --job-name=ldp-omp-regions
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# What an OpenMP region costs on Grace and how many a QP iteration opens, into $OUT/omp_regions.csv:
#   kind=cost:  ranks, threads, n (doubles per rank: FF200's 150,525 dofs over the ranks),
#               empty and sweep (us, region_cost.c)
#   kind=count: FF200 on 1 x 72, the regions the playground opens (region_count.c) over its QP
#               iterations; the few outside the QP are counted too.
# Build both on the login node first, under the same uenv:
#   mpicc -O2 -fopenmp region_cost.c -o region_cost
#   gcc -O2 -shared -fPIC region_count.c -o libregion_count.so -ldl
#   LDP_CASES=... SFEM_BUILD=... OUT=... sbatch omp_regions.sbatch
set -eu
: "${LDP_CASES:?}" "${SFEM_BUILD:?}" "${OUT:?}"
HERE=${SLURM_SUBMIT_DIR:-$(pwd)}
mkdir -p "$OUT"
export OMP_PROC_BIND=true OMP_PLACES=cores MPICH_GPU_SUPPORT_ENABLED=0
CSV=$OUT/omp_regions.csv
echo "kind,ranks,threads,n,empty_us,sweep_us,regions,qp_iterations" > "$CSV"
for shape in "1 72 150525" "4 18 37631" "4 36 37631" "4 72 37631" "16 18 9408"; do
  set -- $shape
  OMP_NUM_THREADS=$2 srun -N 1 -n "$1" --cpus-per-task="$2" --cpu-bind=cores --distribution=block:block \
    "$HERE/region_cost" "$3" | while read -r r t n e s; do echo "cost,$r,$t,$n,$e,$s,,"; done >> "$CSV"
done
D=$OUT/count; rm -rf "$D"; mkdir -p "$D/out"
cd "$LDP_CASES/FF200_ExodusMesh"
# The preload goes to the playground only: srun's own process would count too.
OMP_NUM_THREADS=72 srun -N 1 -n 1 --cpus-per-task=72 --cpu-bind=cores \
  env REGION_COUNT_DIR="$D" LD_PRELOAD="$HERE/libregion_count.so" \
  "$SFEM_BUILD/playground" case.yaml --set output.path="$D/out" > "$D/run.log" 2>&1
its=$(grep -E '^step ' "$D/run.log" | tail -1 | sed -E 's/.*, ([0-9]+) linear.*/\1/')
echo "count,1,72,150525,,,$(cat "$D"/*.txt),$its" >> "$CSV"
cat "$CSV"
