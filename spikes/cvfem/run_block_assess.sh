#!/bin/bash
# Per-block cost of the 2x2 field split, swept over the macro-element level at matched
# problem size (macros * level held constant, so every row solves the same dofs).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
export OMP_PROC_BIND=close OMP_PLACES=cores
export SFEM_BENCH_REPS=${SFEM_BENCH_REPS:-6}
export SFEM_BENCH_VERBOSE_BLOCKS=1
for pair in "2:32" "4:16" "8:8" "16:4"; do
  L=${pair%%:*}; M=${pair##*:}
  echo "### L=$L macros=$M"
  SFEM_BENCH_LEVELS=$L SFEM_BENCH_MACROS=$M ./build/cvfem_sshex8_bench 2>/dev/null \
    | grep -E "ns/dof|% of the full|vs full operator|^[0-9]+ +[0-9]"
done
