#!/usr/bin/env bash
# Compare complete actions while the Krylov direction changes at a fixed state.
set -euo pipefail
bench=${1:?usage: cvfem_jacobian_cache.sh /absolute/path/to/bench}
checker=$(cd "$(dirname "$0")" && pwd)/cvfem_check_dump.sh
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"

check() {
    local layout=$1 threads=$2 label=$3
    shift 3
    local n=10 previous='' arg
    for arg in "$@"; do
        if [[ $previous == --n ]]; then n=$arg; fi
        previous=$arg
    done
    local expected=$((4 * (n + 1) * (n + 1) * (n + 1)))
    for mode in cached per_apply; do
        local extra=(--repeat 3)
        if [[ $mode == per_apply ]]; then extra+=(--rebuild-state-gradients); fi
        if ! OMP_NUM_THREADS=$threads "$bench" --n 10 --pack-size 64 --layout "$layout" \
            --jac-action --warmup 1 --live-vectors 3 --perturb 0.017 \
            --breakdown --dump "$mode.bin" --csv "$mode.csv" "$@" "${extra[@]}" > "$mode.log" 2>&1; then
            cat "$mode.log"
            echo "FAIL: $layout/$threads/$label ($mode)"
            exit 1
        fi
        grep -q "^  state_gradients: $mode$" "$mode.log"
        awk -F, -v mode="$mode" '
            NR == 1 { for (i=1; i<=NF; ++i) if ($i == "jac_state_gradients") col=i; next }
            { if (!col || $col != mode) exit 1; rows++ }
            END { if (!rows) exit 1 }
        ' "$mode.csv"
        # Warmup must not enter a phase fraction whose denominator excludes warmup.
        awk '/^  frac_jac_action_(qgrad|hograd):/ { if ($2 < 0 || $2 > 1.001) exit 1 }' "$mode.log"
        if ! bash "$checker" "$mode.bin" "$expected" > "$mode.txt"; then
            echo "FAIL: $layout/$threads/$label ($mode): invalid output dump"
            exit 1
        fi
        rm "$mode.csv"
    done
    if cmp -s cached.bin per_apply.bin; then
        echo "PASS: $layout/$threads/$label (bit-identical)"
        return
    fi
    # A different field count selects a different compiled reconstruction kernel.
    # Allow 64 scalar rounding units in the infinity norm, not a limiter-sized error.
    local tolerance=1.4210854715202004e-14
    if grep -q '^  scalar: f32' cached.log; then tolerance=7.62939453125e-6; fi
    paste cached.txt per_apply.txt | LC_ALL=C awk -v tolerance="$tolerance" \
        -v label="$layout/$threads/$label" '
        function abs(x) { return x < 0 ? -x : x }
        NF != 2 || toupper($0) ~ /NAN|INF/ { bad=1; next }
        { error=abs($1-$2); if (error>max_error) max_error=error;
          if (abs($1)>scale) scale=abs($1); if (abs($2)>scale) scale=abs($2); count++ }
        END {
            relative=scale ? max_error/scale : max_error;
            ok=!bad && count && relative<=tolerance;
            printf "%s: %s (relative infinity error %.3e)\n", ok ? "PASS" : "FAIL", label, relative;
            exit !ok;
        }'
}

# Packed reconstruction has a fixed sum order at either thread count.
for threads in ${CACHE_TEST_THREADS:-1 2}; do
    check packed "$threads" first_order --rhie-chow
    for limiter in 0 1 2 3; do
        check packed "$threads" "limiter_$limiter" --rhie-chow --conv-ho "$limiter"
        check packed "$threads" "limiter_${limiter}_float_grad" \
            --rhie-chow --conv-ho "$limiter" --grad-precision single
    done
    check packed "$threads" rc_off --conv-ho 0
    check packed "$threads" rc_off_float_grad --conv-ho 0 --grad-precision single
    check packed "$threads" closure --rhie-chow --boundary --transient 0.01 --bdf 2
    check packed "$threads" correction_lagged --rhie-chow --conv-ho 2 --lagged-ho
done
# The unperturbed Venkat case fails the existing layout oracle even before this change.
for limiter in 0 1; do
    check packed 1 "unperturbed_$limiter" --n 24 --rhie-chow --conv-ho "$limiter" --perturb 0
done

# Atomic summation order is repeatable only at one thread.
for layout in atomic ecolor; do
    check "$layout" 1 first_order --rhie-chow
    check "$layout" 1 unlimited --rhie-chow --conv-ho 0
done
check atomic 1 isoparam --rhie-chow --geom isoparam --warp 0.03

if "$bench" --n 2 --rebuild-state-gradients > refused.log 2>&1; then
    echo 'FAIL: accepted state-gradient rebuild for a residual'
    exit 1
fi
grep -q 'requires --jac-action' refused.log
echo 'cvfem_jacobian_cache: PASSED'
