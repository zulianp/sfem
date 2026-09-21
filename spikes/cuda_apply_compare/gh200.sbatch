#!/bin/bash -l
#SBATCH --job-name=cuda_apply_compare
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --output=gh200.log
# The generated kernels, host against device, on one Grace-Hopper node.  Same
# driver source compiled twice, so the only difference between the two runs is
# which generated kernel they link.
set -u
cd $SCRATCH/cuda_apply_compare
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV="uenv run --view=default prgenv-gnu/24.11:v2 --"

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1
lscpu | grep -E "^Model name|^CPU\(s\):"
$UENV nvcc --version | tail -2
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND"
echo

echo "=== throughput, sized until the rate stops moving ==="
for side in 32 64 96 128 160; do
  $UENV ./apply_cpu $side 3 -
  $UENV ./apply_gpu $side 3 -
done
echo

# The agreement cases run on their own small meshes and are recorded only on
# this pass; the side here is the throughput sweep's largest and is incidental.
echo "=== agreement ==="
$UENV ./apply_cpu 32 1 host.txt
$UENV ./apply_gpu 32 1 device.txt
$UENV python3 compare.py host.txt device.txt
echo "=== DONE"
