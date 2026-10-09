#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-fused-rc
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=frc_%j.out
#SBATCH --error=frc_%j.err
# What the packed Jacobian action costs once it carries the term the solver actually runs.
#
# Two questions, in this order because the second is only worth asking if the first
# answers cleanly:
#
#   A. Is the default path -- no --rhie-chow, the configuration every number in
#      perf/baseline_grace.csv was measured in -- unchanged by the new staging? The
#      staging adds one hoisted branch per element batch to a kernel that had a
#      compile-time-null Rhie-Chow pointer before, so this is not free by inspection and
#      has to be measured. bench_before is the binary built from the parent commit.
#
#   B. With Rhie-Chow on, where does the throughput go? The bench has been quoting the
#      element kernel alone; the solver evaluates that kernel plus the direction's nodal
#      gradient reconstruction, which cannot be hoisted out of a Krylov solve because the
#      direction changes every matvec.
#
# --exclusive because a throughput measured on a shared node is not a throughput.
set -u
B=$SCRATCH/cvfem_blk/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_frc/$SLURM_JOB_ID}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT"

run() {  # run <binary> <label> <n> <extra...>
  local bin=$1 label=$2 n=$3; shift 3
  local log="$OUT/$(echo "$label" | tr "/=" "__").log"
  uenv run $U -- $bin --n "$n" --layout packed --jac-action --repeat 20 --warmup 5 "$@" \
    > "$log" 2>&1 || { echo "  $label FAILED"; tail -3 "$log"; return; }
  local dofs mdofs kern frac rel
  dofs=$(grep -m1 -o 'dofs: [0-9]*' "$log" | awk '{print $2}')
  mdofs=$(grep -m1 'MDOF/s_jac_action:' "$log" | awk '{print $2}')
  kern=$(grep -m1 'MDOF/s_jac_action_kernel_only:' "$log" | awk '{print $2}')
  frac=$(grep -m1 'frac_jac_action_qgrad:' "$log" | awk '{print $2}')
  rel=$(grep -m1 'jac_action_rc_vs_atomic_rel:' "$log" | awk '{print $2}')
  printf "%-26s ndof=%-10s MDOF/s=%-10s kernel_only=%-10s qgrad=%-8s vs_atomic=%s\n" \
    "$label" "${dofs:-?}" "${mdofs:-?}" "${kern:--}" "${frac:--}" "${rel:--}"
}

echo
echo "=== A. default path unchanged? (no --rhie-chow, 72 threads) ==="
for n in 60 100 140; do
  run $B/bench_before                   "before_n$n"  $n
  run $B/cvfem_hex8_ns_upwind_bench     "after_n$n"   $n
done

echo
echo "=== B. with fused Rhie-Chow (72 threads) ==="
for n in 60 100 140; do
  run $B/cvfem_hex8_ns_upwind_bench "rc_n$n" $n --rhie-chow 1.0
done

echo
echo "=== C. thread sweep at n=140, with and without Rhie-Chow ==="
for t in 1 8 36 72; do
  export OMP_NUM_THREADS=$t
  run $B/cvfem_hex8_ns_upwind_bench "t${t}_plain" 140
  run $B/cvfem_hex8_ns_upwind_bench "t${t}_rc"    140 --rhie-chow 1.0
done
export OMP_NUM_THREADS=72

echo
echo "done $(date)"
