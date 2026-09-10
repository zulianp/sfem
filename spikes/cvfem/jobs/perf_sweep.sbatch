#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-perf
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=perf_%j.out
#SBATCH --error=perf_%j.err
# Why the same packed sweep runs at 1072 MDOF/s in the solver and 1482 in the benchmark.
#
# Size, the Krylov working set and the element ordering are all settled -- see the
# bench-to-solver section of docs/CVFEM_Throughput.md. What is left splits two ways and
# hardware counters separate them cleanly:
#
#   same instructions, more cycles  ->  the kernel is stalling, and on what
#   more instructions               ->  the two are not running the same code
#
# So: cycles and instructions attributed per symbol, for the element sweep in each
# binary, at matched size (solver 1,048,576 elements x 903 calls, bench 1,000,000 x 20).
# Nothing here is a rate; it is instructions per element visit and IPC.
set -u
B=$SCRATCH/cvfem_blk/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_perf/$SLURM_JOB_ID}; mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)"
perf --version 2>&1 | head -1
echo

# Does perf count at all here? If not, everything below is noise and should say so.
perf stat -e cycles:u,instructions:u -x, true 2>&1 | head -4
echo

# report <perf.data> <symbol-regex> -> "<pct> <total>"
extract() {
  perf report -i "$1" --stdio --no-children -s symbol --percent-limit 0 2>/dev/null | awk -v rx="$2" '
    /Event count/ { gsub(/[^0-9]/, "", $NF); total=$NF }
    $0 ~ rx && /%/ { p=$1; gsub(/%/, "", p); sum += p }
    END { printf "%.3f %s", sum+0, total+0 }'
}

# perf goes INSIDE the uenv, not around it. Wrapping `uenv run` profiles the launcher and
# loses the real process at exec: the first attempt attributed 99.99% of its samples to
# ld-linux-aarch64.so.1 and collected 1.6M cycles for a run that takes seconds.
profile() {  # profile <tag> <symbol-regex> <binary> <args...>
  local tag=$1 rx=$2; shift 2
  for ev in cycles instructions; do
    uenv run $U -- /usr/bin/perf record -q -e ${ev}:u -F 499 -o "$OUT/$tag.$ev.data" \
      -- "$@" > "$OUT/$tag.$ev.log" 2>&1 || true
  done
  local c i
  c=$(extract "$OUT/$tag.cycles.data" "$rx")
  i=$(extract "$OUT/$tag.instructions.data" "$rx")
  echo "$tag cycles: $c"
  echo "$tag instructions: $i"
}

echo "=== benchmark, --n 100 (1,000,000 elements x 20 applies) ==="
( cd "$OUT" && profile bench "apply_jacobian_action_packed" \
    $B/cvfem_hex8_ns_upwind_bench --n 100 --layout packed --jac-action \
      --rhie-chow 1.0 --repeat 20 --warmup 0 )

echo
echo "=== solver, flat N=64 (1,048,576 elements x ~903 applies) ==="
d="$OUT/solve"; mkdir -p "$d"
( cd "$d"
  export SFEM_CASE=poiseuille SFEM_N=64 SFEM_GMG=0 SFEM_PRECOND=bjacobi
  export SFEM_NL_MAX_IT=3 SFEM_LSOLVE_MAX_IT=150 SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0
  profile solver "cvfem_hex8_apply_jacobian_action_packed" \
    $B/cvfem_hex8_ns_ssgmg "$d/out" )
grep -m1 -o 'ndof: [0-9]*' "$d/run.log" 2>/dev/null || true
awk -F, '$1 ~ /apply_jacobian_action_packed/ {printf "solver sweep calls=%s seconds=%s\n", $2, $3}' \
  "$d/smesh.trace.csv" 2>/dev/null || true

echo
for tag in bench solver; do
  dso=$(perf report -i "$OUT/$tag.cycles.data" --stdio --no-children -s dso --percent-limit 50 2>/dev/null \
        | grep '%' | head -1)
  case "$dso" in
    *ld-linux*|"") echo "WARNING: $tag profile did not follow the process ($dso) -- its numbers are meaningless" ;;
  esac
done

echo
echo "top symbols, benchmark (cycles):"
perf report -i "$OUT/bench.cycles.data" --stdio --no-children -s symbol --percent-limit 2 2>/dev/null | grep '%' | head -8
echo
echo "top symbols, solver (cycles):"
perf report -i "$OUT/solver.cycles.data" --stdio --no-children -s symbol --percent-limit 2 2>/dev/null | grep '%' | head -10

echo
echo "done $(date)"
