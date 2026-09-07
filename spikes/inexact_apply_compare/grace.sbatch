# The split benchmark on one socket of an Alps GH200 node.
#
#   sbatch grace.sbatch     (from the directory holding bench_mixed_grace)
#
# The u2 account was out of node-minutes when this was first run; c40 was not.
# Build first, inside the uenv, since the system g++ is 7.5:
#
#   uenv run --view=default prgenv-gnu/24.11:v2 -- mpic++ -std=c++17 -O3 \
#       -mcpu=native -DNDEBUG -fopenmp -o bench_mixed_grace ...
#
# The generated tree must keep its directory structure: the emitted headers
# include across it (../../kernel_math.hpp).
#!/bin/bash -l
#SBATCH --job-name=split_bench
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --output=grace_bench.log
cd $SCRATCH/grace_tree/d3/tet4
echo "=== node ==="
hostname
lscpu | grep -E "^Model name|^Socket|^Core|^CPU\(s\):|^Thread|^NUMA node\(s\)"
echo
for T in 1 2 4 8 16 32 64 72; do
  echo "--- OMP_NUM_THREADS=$T ---"
  OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES=cores     uenv run --view=default prgenv-gnu/24.11:v2 -- ./bench_mixed_grace 20 2>&1     | sed -n "1p;/384000/p;/rel diff/p;/break-even/p"
done
echo "=== DONE ==="
