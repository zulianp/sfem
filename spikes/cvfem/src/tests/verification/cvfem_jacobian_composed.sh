#!/usr/bin/env bash
# Check the composed residual, keeping BDF history fixed during differentiation.
set -euo pipefail
bench=${1:?usage: cvfem_jacobian_composed.sh /absolute/path/to/bench}
work=$(mktemp -d)
complete=0
trap 'rc=$?; rm -rf "$work"; if [[ $complete != 1 && $rc == 0 ]]; then rc=1; fi; exit "$rc"' EXIT
cd "$work"
args=(--n 8 --pack-size 64 --perturb 0.017 --layout packed --rhie-chow
      --rho 1.3 --mu 0.027 --verify-jac-coupled)
for threads in ${COUPLED_TEST_THREADS:-1 2}; do
    checked=0
    for rc_time in 0 1; do
        for mode in boundary bdf1 bdf2 boundary-bdf1 boundary-bdf2; do
            [[ $rc_time == 1 && $mode == boundary ]] && continue
            closure=() boundary=0 order=0 dt=0 inv_dt=0
            if [[ $mode == boundary* ]]; then closure+=(--boundary); boundary=1; fi
            if [[ $mode == *bdf1 ]]; then order=1; inv_dt=10; fi
            if [[ $mode == *bdf2 ]]; then order=2; inv_dt=15; fi
            if [[ $order != 0 ]]; then closure+=(--transient 0.1 --bdf "$order"); dt=0.1; fi
            [[ $rc_time == 0 ]] && inv_dt=0
            for scheme in first 0 1 2 3; do
                flags=(--jac-action)
                if [[ $scheme != first ]]; then flags+=(--conv-ho "$scheme"); fi
                if ! OMP_NUM_THREADS=$threads SFEM_RC_TAU=1 SFEM_RC_TAU_TRANSIENT=$rc_time \
                    "$bench" "${args[@]}" "${closure[@]}" "${flags[@]}" > check.log 2>&1; then
                    cat check.log
                    exit 1
                fi
                echo "threads=$threads scheme=$scheme mode=$mode rc_time=$rc_time"
                grep '^verify_jac_coupled' check.log
                grep -q "^verify_jac_coupled_scope: boundary $boundary bdf $order dt $dt rc_inv_dt_a0 $inv_dt threads $threads$" check.log
                grep -q '^verify_jac_coupled: PASSED' check.log
                test "$(grep -c 'required 1' check.log)" -eq 2
                checked=$((checked + 1))
            done
        done
    done
    test "$checked" -eq 45
done
complete=1
echo 'cvfem_jacobian_composed: PASSED'
