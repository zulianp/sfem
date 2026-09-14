#!/bin/bash -l
#SBATCH --job-name=cuda_a2
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --output=$SCRATCH/cuda_apply_compare/a2.log
# The generated neohookean PROTEUS_HEX8 matrix-free apply, host against device,
# on one Grace-Hopper node.  Same driver source compiled twice, so the only
# difference between the two runs is which generated kernel they link.
set -u
cd $SCRATCH/cuda_apply_compare
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
echo "=== node ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
lscpu | grep -E "^Model name|^CPU\(s\):"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND"
echo
for side in 32 64 96 128; do
  echo "--- side $side ---"
  uenv run --view=default prgenv-gnu/24.11:v2 -- ./apply_cpu $side 3 -
  uenv run --view=default prgenv-gnu/24.11:v2 -- ./apply_gpu $side 3 -
done
echo "=== DONE"
