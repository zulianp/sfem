#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-perfstat
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=pstat_%j.out
#SBATCH --error=pstat_%j.err
# Exact totals, to turn the sampled per-symbol percentages into absolutes.
#
# perf record in frequency mode reports an "Event count (approx.)" that is a sum of
# sample periods, and using it as an absolute gave a bench sweep consuming more
# thread-cycles than the wall clock had available -- impossible, and a reminder that the
# percentages are the trustworthy part of a sampled profile. perf stat counts rather than
# samples, so totals from here multiplied by those percentages are sound.
#
# Also counts the ARM cache and stall events, which say WHY if the instruction counts
# turn out to match. Each event group is run separately so one unsupported name cannot
# take the rest of the run down with it.
set -u
B=$SCRATCH/cvfem_blk/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_pstat/$SLURM_JOB_ID}; mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"

stat_run() {  # stat_run <tag> <events> <binary> <args...>
  local tag=$1 ev=$2; shift 2
  uenv run $U -- /usr/bin/perf stat -e "$ev" -x, -o "$OUT/$tag.csv" -- "$@" > "$OUT/$tag.log" 2>&1 || true
  echo "--- $tag ($ev) ---"
  grep -vE '^#|^$' "$OUT/$tag.csv" 2>/dev/null | awk -F, '{printf "  %-28s %s\n", $3, $1}'
}

echo
echo "=== benchmark: --n 100, 1,000,000 elements x 20 applies ==="
( cd "$OUT" && stat_run bench_base "cycles:u,instructions:u,task-clock" \
    $B/cvfem_hex8_ns_upwind_bench --n 100 --layout packed --jac-action --rhie-chow 1.0 --repeat 20 --warmup 0 )
( cd "$OUT" && stat_run bench_mem "L1D_CACHE_REFILL:u,L2D_CACHE_REFILL:u,LL_CACHE_MISS_RD:u,MEM_ACCESS:u" \
    $B/cvfem_hex8_ns_upwind_bench --n 100 --layout packed --jac-action --rhie-chow 1.0 --repeat 20 --warmup 0 )
( cd "$OUT" && stat_run bench_stall "STALL_FRONTEND:u,STALL_BACKEND:u" \
    $B/cvfem_hex8_ns_upwind_bench --n 100 --layout packed --jac-action --rhie-chow 1.0 --repeat 20 --warmup 0 )

echo
echo "=== solver: flat N=64, 1,048,576 elements x 903 applies ==="
d="$OUT/solve"; mkdir -p "$d"
run_solver() { ( cd "$d"
  export SFEM_CASE=poiseuille SFEM_N=64 SFEM_GMG=0 SFEM_PRECOND=bjacobi
  export SFEM_NL_MAX_IT=3 SFEM_LSOLVE_MAX_IT=150 SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0
  stat_run "$1" "$2" $B/cvfem_hex8_ns_ssgmg "$d/out" ) ; }
run_solver solver_base  "cycles:u,instructions:u,task-clock"
run_solver solver_mem   "L1D_CACHE_REFILL:u,L2D_CACHE_REFILL:u,LL_CACHE_MISS_RD:u,MEM_ACCESS:u"
run_solver solver_stall "STALL_FRONTEND:u,STALL_BACKEND:u"
awk -F, '$1 ~ /apply_jacobian_action_packed/ {printf "  solver sweep: calls=%s seconds=%s\n", $2, $3}' \
  "$d/smesh.trace.csv" 2>/dev/null || true

echo
echo "done $(date)"
