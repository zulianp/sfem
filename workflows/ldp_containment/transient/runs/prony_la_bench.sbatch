#!/bin/bash
#SBATCH --job-name=prony-la-bench
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/prony_la_bench_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The history formats' bytes, step cost and error (drivers/bench/bench_prony_history: its synthetic
# stress and its built-in series, four stored terms at 0.1 ms), without and with the remainder form
# (horizon 0.1655 s, LDP24A's strokes: two of the four stored terms): Grace (72 threads, bound) and one
# H100. Shapes: LDP24A's TET4 (742,862 one-point elements), eight points (HEX8), eleven (TET10) and
# 64 (HEX27). The output is this job's log. The machine: ../../machine.sh.
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
B=${SFEM_BUILD:-$SFEM_ROOT/build-cuda}/bench_prony_history
for space in host device; do
  for shape in "742862 1 30 16" "250000 8 30 0" "250000 11 30 0" "40000 64 30 0"; do
    for horizon in 0 0.1655; do
      echo "## $space $shape $horizon"
      SFEM_EXECUTION_SPACE=$space $SFEM_LAUNCH_GPU $B $shape $horizon 2>&1 | grep -E "^format|,host,|,device,|rror"
    done
  done
done
