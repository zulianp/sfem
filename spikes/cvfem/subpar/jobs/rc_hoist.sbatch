#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-rc-hoist
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=rch_%j.out
#SBATCH --error=rch_%j.err
# The hoisted Rhie-Chow coefficient, against the version that computed it in the face loop.
#
# bench_A is built from the parent commit; the unsuffixed binary is the change. Three
# things have to hold at once and each is a separate section below:
#
#   1. the packed Jacobian action gets materially faster with Rhie-Chow on;
#   2. the residual, which runs the same coefficient in the other face loop, does too;
#   3. nothing moves when Rhie-Chow is off -- that is every number in
#      perf/baseline_grace.csv.
#
# The packed result is checked against the atomic layout inside the driver on every run
# (jac_action_rc_vs_atomic_rel), and the atomic layout still computes the coefficient
# in-kernel from the same function, so that comparison is exactly the assertion that
# hoisting did not change the value.
#
# Best of five because the run-to-run spread on this node is about 3% and the residual
# difference this has to resolve is smaller than that.
set -u
B=$SCRATCH/cvfem_blk/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_rch/$SLURM_JOB_ID}; mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"

run() {  # run <binary> <label> <extra...>
  local bin=$1 label=$2; shift 2
  local best="" cs="" ko="" rel="" dofs=""
  for r in 1 2 3 4 5; do
    local log="$OUT/${label}_r$r.log"
    uenv run $U -- "$B/$bin" --n 140 --layout packed --repeat 20 --warmup 5 "$@" \
      > "$log" 2>&1 || { echo "  $label FAILED"; tail -3 "$log"; return; }
    local m; m=$(grep -m1 '^  MDOF/s:' "$log" | awk '{print $2}')
    best=$(awk -v a="$best" -v b="$m" 'BEGIN{print (a=="" || b+0>a+0)?b:a}')
    dofs=$(grep -m1 -o 'dofs: [0-9]*' "$log" | awk '{print $2}')
    ko=$(grep -m1 'MDOF/s_jac_action_kernel_only:' "$log" | awk '{print $2}')
    rel=$(grep -m1 'jac_action_rc_vs_atomic_rel:' "$log" | awk '{print $2}')
    cs=$(grep -m1 'checksum:' "$log" | awk '{print $2}')
  done
  printf "  %-22s ndof=%-10s MDOF/s=%-10s kernel_only=%-10s vs_atomic=%-14s checksum=%s\n" \
    "$label" "$dofs" "$best" "${ko:--}" "${rel:--}" "$cs"
}

echo
echo "=== 1. packed Jacobian action, WITH Rhie-Chow ==="
run bench_A                       "before" --jac-action --rhie-chow 1.0
run cvfem_hex8_ns_upwind_bench    "after"  --jac-action --rhie-chow 1.0
echo
echo "=== 2. packed residual, WITH Rhie-Chow ==="
run bench_A                       "res_before" --rhie-chow 1.0
run cvfem_hex8_ns_upwind_bench    "res_after"  --rhie-chow 1.0
echo
echo "=== 3. no Rhie-Chow: must not move (the baseline configuration) ==="
run bench_A                       "plain_jac_before" --jac-action
run cvfem_hex8_ns_upwind_bench    "plain_jac_after"  --jac-action
run bench_A                       "plain_res_before"
run cvfem_hex8_ns_upwind_bench    "plain_res_after"

echo
echo "=== 4. the solver: same iterates? (Poiseuille N=48, bjacobi, no MG) ==="
for v in ssgmg_A cvfem_hex8_ns_ssgmg; do
  d="$OUT/solve_$v"; mkdir -p "$d"
  ( cd "$d"
    export SFEM_CASE=poiseuille SFEM_N=48 SFEM_GMG=0 SFEM_PRECOND=bjacobi
    export SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0 SFEM_NL_MAX_IT=8
    uenv run $U -- "$B/$v" "$d/out" > run.log 2>&1 )
  printf "  %-22s %s\n" "$v" \
    "$(grep -hoE 'ndof: [0-9]+|newton_total: [0-9]+|lin_it_total: [0-9]+|t_solve: [0-9.]+ s|u_linf: [0-9.e+-]+' "$d/run.log" | tr '\n' ' ')"
done

echo
echo "done $(date)"
