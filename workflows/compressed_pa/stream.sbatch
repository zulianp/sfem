#!/bin/bash
#SBATCH --job-name=cpa-stream
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The STREAM triad of the compressed partial assembly paper's bandwidth bound (stream_triad.cpp): one Grace
# chip with 72 threads, as the benchmarks run, and one H100. Three runs each.
#   [SFEM_ROOT=<tree>] [OUT=<a new file>]
set -u
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
W=$SFEM_ROOT/workflows/compressed_pa
T=$(mktemp -d)
g++ -O3 -mcpu=neoverse-v2 -fopenmp -o "$T/stream_host" "$W/stream_triad.cpp" || exit 1
nvcc -O3 -arch=sm_90 -x cu -o "$T/stream_device" "$W/stream_triad.cpp" || exit 1
for r in 1 2 3; do
  OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores srun -n 1 -c 72 --cpu-bind=none numactl --cpunodebind=0 --membind=0 "$T/stream_host"
  srun -n 1 -c 72 --gpus=1 "$T/stream_device"
done
