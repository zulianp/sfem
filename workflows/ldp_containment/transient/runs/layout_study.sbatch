#!/bin/bash
#SBATCH --job-name=qbsrlayout
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=00:50:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/layout_study_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The quantized BSR's device layout: SELL-32 thread per row against CSR order with G = 4, 8, 16 threads
# per row (the BSR group kernel's shape), on the FF200 valve meshes and on cubes, twice each in
# alternating order.
set -u
C=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh
B=${B:-/ritom/scratch/cscs/zulianp/sfem-prony-md/build-cuda-s2b}
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores SFEM_EXECUTION_SPACE=device
for shape in "MESH $C/tp9_p2_bsr32__full_a/sim2/mesh" "MESH $C/tp9_s0_bsr32__full_a/sim2/mesh" "TET10 34" "TET4 68" "TET10 74"; do
  for rep in 1 2; do
    for cfg in sell csr:16 csr:8 csr:4; do
      layout=${cfg%%:*}
      g=16; [ "$layout" = csr ] && g=${cfg#*:}
      echo "== $shape rep$rep $cfg"
      # shellcheck disable=SC2086
      SFEM_QBSR_LAYOUT=$layout SFEM_QBSR_G=$g srun -n 1 -c 72 --gpus-per-task=1 $B/bench_inexact $shape 50 2>&1 | grep -E "^(BSR f32|bsr)"
    done
  done
done
