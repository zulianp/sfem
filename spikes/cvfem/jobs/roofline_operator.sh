#!/usr/bin/env bash
# Run inside the user's exclusive one-socket allocation after building the benchmark.
set -euo pipefail
bench=${1:?usage: roofline_operator.sh benchmark /new/results/directory}
out=${2:?usage: roofline_operator.sh benchmark /new/results/directory}
[[ $# == 2 && ${SLURM_CPUS_PER_TASK:-0} == 72 ]] || { echo 'Request 72 CPU cores'; exit 1; }
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir "$out"
out=$(cd "$out" && pwd)
exec > >(tee "$out/run.log") 2>&1
# Both references and operators must pass in this allocation with the same binding.
bash "$root/jobs/roofline_calibration.sh" "$out/calibration" --validate-reuse
python3 "$root/scripts/roofline_operator_test.py" "$bench"
cp "$root/scripts/roofline_operator.py" "$root/scripts/roofline_operator_test.py" \
    "$root/scripts/roofline_reuse_diagnostic.py" "$root/jobs/roofline_operator.sh" \
    "$root/src/drivers/cvfem_hex8_ns_upwind_bench.cpp" "$root/src/drivers/cvfem_perf_control.hpp" \
    "$root/src/tests/verification/cvfem_roofline_window.hpp" "$out/"
sha256sum "$root/scripts/roofline_operator.py" "$root/scripts/roofline_operator_test.py" \
    "$root/jobs/roofline_operator.sh" "$root/src/drivers/cvfem_hex8_ns_upwind_bench.cpp" \
    "$root/src/drivers/cvfem_perf_control.hpp" "$root/src/tests/verification/cvfem_roofline_window.hpp" \
    "$root/scripts/roofline_reuse_diagnostic.py" > "$out/operator-source.sha256"
python3 "$root/scripts/roofline_operator.py" --binary "$bench" --out "$out/operators" --calibration "$out/calibration/captures"
echo "ROOFLINE RESIDUAL STUDY PASSED: $out"
