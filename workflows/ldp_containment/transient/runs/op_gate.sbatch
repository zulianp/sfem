#!/bin/bash
#SBATCH --job-name=opgate
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/op_gate_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The operator gate of the stored tangents (P-op): bench_inexact on one H100, TET4 and TET10 cubes at
# about 1M and 10M dofs, A B B A: build A's bench_inexact against build B's (B unset: A again).
#   A=<build dir> [B=<build dir>] [MATERIAL=GeneratedNeoHookeanOgden] [REPEATS=50]
set -u
B=${B:-$A}
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores SFEM_EXECUTION_SPACE=device
for shape in "TET4 68" "TET4 149" "TET10 34" "TET10 74"; do
  set -- $shape
  for slot in A1 B1 B2 A2; do
    dir=$A; [ ${slot:0:1} = B ] && dir=$B
    echo "== $slot $dir $1 $2"
    srun --exact -n 1 -c 72 --gpus-per-task=1 $dir/bench_inexact $1 $2 ${REPEATS:-50} ${MATERIAL:-GeneratedNeoHookeanOgden} 2>&1 | tail -n +2
  done
done
