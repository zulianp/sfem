#!/bin/bash
#SBATCH --job-name=qbsrwin
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/window_study_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The quantized BSR's windowed update: set-up time against the scratch it holds while it runs, per
# number of windows (SFEM_QBSR_WINDOWS), on the FF200 valve meshes, twice each.
set -u
B=${B:-/ritom/scratch/cscs/zulianp/sfem-s4hand/build-cuda}
C=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores SFEM_EXECUTION_SPACE=device
for m in tp9_p2_bsr32__full_a tp9_s0_bsr32__full_a; do
  for rep in 1 2; do
    for w in 1 2 4 8; do
      echo "== $m rep$rep windows=$w"
      SFEM_QBSR_WINDOWS=$w srun -n 1 -c 72 --gpus-per-task=1 $B/bench_inexact MESH $C/$m/sim2/mesh 20 2>&1 | grep -E "^bsr"
    done
  done
done
