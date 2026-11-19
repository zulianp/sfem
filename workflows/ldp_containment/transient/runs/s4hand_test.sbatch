#!/bin/bash
#SBATCH --job-name=s4hand
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/s4hand_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The windowed quantized-BSR update, hand-edited generated kernels (sfem-s4hand): the device store
# tests on TET4/TET10, then bench_inexact on the FF200 valve meshes with one window and the default.
set -u
B=${B:-/ritom/scratch/cscs/zulianp/sfem-s4hand/build-cuda}
C=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores SFEM_EXECUTION_SPACE=device
echo "== sfem_CudaInexactOperatorTest"
srun -n 1 -c 72 --gpus-per-task=1 $B/sfem_CudaInexactOperatorTest 2>&1 | grep -E "^(TET4|TET10) .*(bsr|fused|window)|failed|FAIL|Assertion|passed" | head -40
for m in tp9_s0_bsr32__full_a tp9_p2_bsr32__full_a; do
  for w in 1 default; do
    echo "== $m windows=$w"
    if [ $w = default ]; then unset SFEM_QBSR_WINDOWS; else export SFEM_QBSR_WINDOWS=$w; fi
    srun -n 1 -c 72 --gpus-per-task=1 $B/bench_inexact MESH $C/$m/sim2/mesh 50 2>&1 | grep -E "operator|^(BSR f32|bsr)"
  done
done
