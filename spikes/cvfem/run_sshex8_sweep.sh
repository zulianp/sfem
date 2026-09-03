#!/bin/bash
# Level sweep at matched problem size: macros * level is held constant, so every row
# solves the same number of dofs and only the macro-element size changes.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
export OMP_PROC_BIND=close OMP_PLACES=cores
export SFEM_BENCH_REPS=${SFEM_BENCH_REPS:-5}
echo "L    ndof       naive_ns/d   macro_ns/d   affine_ns/d  hoist_ns/d   sp_hoist  best_MDOF/s  agree_rel"
for pair in "2:32" "4:16" "8:8" "16:4"; do
  L=${pair%%:*}; M=${pair##*:}
  SFEM_BENCH_LEVELS=$L SFEM_BENCH_MACROS=$M ./build/cvfem_sshex8_bench 2>/dev/null | grep -E "^[0-9]+ +[0-9]" | tail -1
done
