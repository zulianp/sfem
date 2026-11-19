#!/usr/bin/env bash
# Check all four equations and expose both Rhie-Chow sensitivities.
set -euo pipefail
bench=${1:?usage: cvfem_jacobian_coupled.sh /absolute/path/to/bench}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
args=(--n 8 --pack-size 64 --perturb 0.017 --layout packed --rhie-chow --verify-jac-coupled)
for threads in ${COUPLED_TEST_THREADS:-1 2}; do
    for scheme in first 0 1 2 3; do
        flags=(--jac-action)
        if [[ $scheme != first ]]; then flags+=(--conv-ho "$scheme"); fi
        if ! OMP_NUM_THREADS=$threads "$bench" "${args[@]}" "${flags[@]}" > check.log 2>&1; then
            cat check.log
            exit 1
        fi
        echo "threads=$threads scheme=$scheme"
        grep '^verify_jac_coupled:' check.log
        grep -q '^verify_jac_coupled: PASSED' check.log
        test "$(grep -c 'required 1' check.log)" -eq 2
    done
done
for option in '--lagged-rc' '--lagged-ho' '--boundary' '--transient 0.1' \
              '--geom isoparam' '--grad-precision single' '--layout atomic'; do
    if OMP_NUM_THREADS=1 "$bench" "${args[@]}" $option > refused.log 2>&1; then
        echo "FAIL: coupled check accepted unsupported option $option"
        exit 1
    fi
    grep -q 'requires a standalone packed affine steady RC action' refused.log
done
if OMP_NUM_THREADS=1 SFEM_RC_TAU=0 "$bench" "${args[@]}" > refused.log 2>&1; then
    echo 'FAIL: coupled check accepted a velocity-independent RC coefficient'
    exit 1
fi
grep -q 'requires the default steady RC coefficient' refused.log
echo 'cvfem_jacobian_coupled: PASSED'
