#!/usr/bin/env bash
# A timing flag must reach the operation it names.
set -euo pipefail
bench=${1:?usage: cvfem_kernel_only.sh /absolute/path/to/bench}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export OMP_NUM_THREADS=1
args=(--n 2 --pack-size 8 --repeat 1 --warmup 0)
refuse() {
    local status=0
    "$bench" "${args[@]}" --kernel-only "$@" > check.log 2>&1 || status=$?
    if [[ $status != 2 ]] || ! grep -q -- '--kernel-only requires standalone --assemble' check.log; then
        cat check.log
        echo "FAIL: expected a kernel-only refusal for $*"
        exit 1
    fi
    if grep -q 'MDOF/s:' check.log; then
        echo 'FAIL: refused configuration reported a timing'
        exit 1
    fi
}
for layout in packed atomic colored ecolor store; do
    refuse --layout "$layout" --rhie-chow
    refuse --layout "$layout" --rhie-chow --jac-action
done
refuse --layout atomic --assemble
refuse --layout colored --assemble
refuse --layout ecolor --assemble
refuse --layout store --geom isoparam --assemble
refuse --layout atomic --assemble-diag
refuse --layout packed --bsr-apply
refuse --layout packed --nodal-grad 1
refuse --layout packed --assemble --verify-jac
refuse --layout packed --assemble --mesh-footprint
for geometry in affine isoparam; do
    "$bench" "${args[@]}" --kernel-only --assemble --layout packed --geom "$geometry" > check.log 2>&1
    grep -q 'seconds_per_assemble:' check.log
done
"$bench" "${args[@]}" --kernel-only --assemble --layout store > check.log 2>&1
grep -q 'seconds_per_assemble:' check.log
for layout in packed atomic; do
    "$bench" "${args[@]}" --layout "$layout" --rhie-chow > check.log 2>&1
    grep -q 'MDOF/s:' check.log
done
echo 'cvfem_kernel_only: PASSED (19 refusals, 3 assembly diagnostics, 2 residual controls)'
