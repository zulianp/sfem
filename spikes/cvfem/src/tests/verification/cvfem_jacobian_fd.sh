#!/usr/bin/env bash
# FP64 velocity-direction check on affine geometry, without Rhie-Chow coupling.
set -euo pipefail
bench=${1:?usage: cvfem_jacobian_fd.sh /absolute/path/to/bench}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
args=(--n 8 --pack-size 64 --perturb 0.017 --layout packed --jac-action
      --verify-jac-ho --repeat 1 --warmup 0)
for threads in ${FD_TEST_THREADS:-1 2}; do
    for eps in 1e-7 1e-8; do
        for limiter in 0 1 2 3; do
            if ! OMP_NUM_THREADS=$threads CVFEM_FD_EPS=$eps "$bench" "${args[@]}" \
                    --conv-ho "$limiter" > check.log 2>&1; then
                cat check.log
                exit 1
            fi
            grep '^verify_jac_ho:' check.log
        done
    done
done
# The larger step crosses clip branches. The old percentage rule accepted it.
if OMP_NUM_THREADS=1 CVFEM_FD_EPS=1e-6 "$bench" "${args[@]}" --conv-ho 1 > crossing.log 2>&1; then
    cat crossing.log
    echo 'FAIL: accepted the branch-crossing finite difference'
    exit 1
fi
grep -q 'FAIL: exact higher-order action differs' crossing.log
for eps in 0 -1 nan inf; do
    if OMP_NUM_THREADS=1 CVFEM_FD_EPS=$eps "$bench" "${args[@]}" --conv-ho 0 > invalid.log 2>&1; then
        echo "FAIL: accepted CVFEM_FD_EPS=$eps"
        exit 1
    fi
    grep -q 'FAIL: CVFEM_FD_EPS must be finite and positive' invalid.log
done
echo 'cvfem_jacobian_fd: PASSED'
