#!/usr/bin/env bash
set -euo pipefail
bench=${1:?usage: cvfem_scatter_study.sh /absolute/path/to/bench}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export OMP_DYNAMIC=FALSE
args=(--n 5 --pack-size 64 --repeat 1 --warmup 1 --scatter-samples 6 --scatter-study --perturb .017)
for threads in ${SCATTER_TEST_THREADS:-1 2}; do
    export OMP_NUM_THREADS=$threads
    for layout in packed atomic; do
        for scheme in fo 0 1 2 3; do
            case_args=("${args[@]}" --layout "$layout" --rhie-chow)
            [[ $scheme == fo ]] || case_args+=(--conv-ho "$scheme")
            "$bench" "${case_args[@]}" > check.log 2>&1 || { cat check.log; exit 1; }
            grep -q '^scatter_gate PASS$' check.log
            grep -q '^scatter_study PASS$' check.log
            [[ $(grep -c '^scatter_sample ' check.log) == 6 ]]
        done
        "$bench" "${args[@]}" --layout "$layout" > check.log 2>&1 || { cat check.log; exit 1; }
        grep -q '^scatter_study PASS$' check.log
    done
done
export OMP_NUM_THREADS=1
refuse() {
    local status=0
    "$bench" "${args[@]}" "$@" > check.log 2>&1 || status=$?
    [[ $status != 0 ]] || { cat check.log; echo "FAIL: accepted $*"; exit 1; }
    if grep -q '^scatter_sample ' check.log; then cat check.log; exit 1; fi
}
refuse --assemble
refuse --jac-action
refuse --geom isoparam
refuse --layout store
refuse --boundary
refuse --dt .1
refuse --grad-precision single
refuse --pgrad-per-apply
refuse --kernel-only
refuse --breakdown
refuse --warmup 0
refuse --scatter-samples 7
echo 'cvfem_scatter_study: PASSED'
