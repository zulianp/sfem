#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-bench-kernels
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=bench_solve_kernels_%j.out
#SBATCH --error=bench_solve_kernels_%j.err
#
# The determinism study, repeated on the benchmark, and the dof rate of every kernel the
# solve calls. One allocation, one idle socket: the same binary reads 5-11% apart on two
# different Grace nodes, so rows gathered from separate jobs would be reporting the machine.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
B=$ROOT/build_solver/cvfem_hex8_ns_upwind_bench
OUT=${OUT:-$SCRATCH/cvfem_bench_solve_kernels/${SLURM_JOB_ID:-manual}}
CSV=$OUT/kernels.csv
mkdir -p "$OUT"
export OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
BIG=${BIG:-100}   # 4,121,204 dof, past saturation
echo "host $(hostname)  $(date)"
echo "csv  $CSV"

# ---- part 1: is the bench kernel bit-reproducible across thread counts? ----
#
# The checksum the bench prints is summed serially over the result, so two runs agreeing on
# all sixteen digits means the result arrays agree. Compared across thread counts, with the
# boundary closure gathered (default) and scattered (SFEM_BND_ATOMIC=1).
echo
echo "=== determinism: checksum against thread count, 4,121,204 dof ==="
printf "%-22s %-8s %-5s %s\n" config mode thr checksum
det() {
    local tag=$1; shift
    for mode in gather atomic; do
        local A=$( [ $mode = atomic ] && echo 1 || echo 0 )
        for t in 1 8 36 72; do
            local log=$OUT/det_${tag}_${mode}_$t.log
            uenv run $U -- env OMP_NUM_THREADS=$t OMP_PROC_BIND=true OMP_PLACES=cores \
                SFEM_BND_ATOMIC=$A "$B" --n $BIG --repeat 1 --warmup 0 "$@" > "$log" 2>&1
            printf "%-22s %-8s %-5s %s\n" "$tag" "$mode" "$t" \
                "$(grep -m1 '  checksum:' "$log" | awk '{print $2}')"
        done
    done
}
det residual_solver --layout packed --rhie-chow --pgrad-per-apply --boundary
det jac_solver      --layout packed --jac-action --rhie-chow --boundary

# ---- part 2: the dof rate of each kernel the solve calls ----
echo
echo "=== throughput, 4,121,204 dof, 72 threads, 10 timed applies after 5 warmups ==="
export OMP_NUM_THREADS=72
printf "%-24s %12s %10s\n" kernel MDOF/s ms/call
run() {
    local tag=$1; shift
    local log=$OUT/$tag.log
    if uenv run $U -- "$B" --n $BIG --repeat 10 --warmup 5 --breakdown \
            --csv "$CSV" --tag "$tag" "$@" > "$log" 2>&1; then
        printf "%-24s %12s %10s\n" "$tag" \
            "$(grep -m1 '  MDOF/s:' "$log" | awk '{printf "%.1f", $2}')" \
            "$(grep -m1 '  seconds per call:' "$log" | awk '{printf "%.3f", $4*1000}')"
    else
        printf "%-24s %12s\n" "$tag" "FAILED"
    fi
}
# The matvec the Krylov loop applies, built up one term at a time so each stage's cost is a
# difference of two measured rows rather than an attribution.
run jac_bare        --layout packed --jac-action
run jac_rc          --layout packed --jac-action --rhie-chow
run jac_solver      --layout packed --jac-action --rhie-chow --boundary
run jac_solver_dt   --layout packed --jac-action --rhie-chow --boundary --transient 0.01
# The residual, as the Newton step evaluates it.
run res_bare        --layout packed
run res_rc          --layout packed --rhie-chow
run res_solver      --layout packed --rhie-chow --pgrad-per-apply --boundary
# What the preconditioner is built from and what the coarse levels apply.
run diag_solver     --layout atomic --assemble-diag --rhie-chow --boundary
run asm_solver      --layout packed --assemble --rhie-chow --boundary
run spmv            --layout packed --bsr-apply

echo
echo "=== per-phase breakdown of the solver's matvec ==="
grep -A12 'phase' $OUT/jac_solver.log | head -16
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
