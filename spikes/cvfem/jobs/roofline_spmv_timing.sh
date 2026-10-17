#!/usr/bin/env bash
# This collects timing diagnostics and cannot accept a performance campaign.
set -euo pipefail
bench=${1:?usage: roofline_spmv_timing.sh benchmark /new/results/directory}
out=${2:?missing results directory}
option=${3:-}
[[ ( $# == 2 || ( $# == 3 && ( $option == --resources || $option == --thp-study ) ) ) && ${SLURM_CPUS_PER_TASK:-0} == 72 ]] || { echo 'Request 72 CPU cores; optional flag is --resources or --thp-study'; exit 1; }
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir "$out"
out=$(cd "$out" && pwd)
exec > >(tee "$out/run.log") 2>&1
python3 "$root/scripts/roofline_spmv_timing_test.py" "$bench"
if [[ $option == --thp-study ]]; then python3 "$root/scripts/roofline_thp_study_test.py" "$bench"; fi
sources=("$root/scripts/roofline_spmv_timing.py" "$root/scripts/roofline_spmv_timing_test.py"
    "$root/scripts/roofline_process_observation.py"
    "$root/scripts/roofline_thp_study.py" "$root/scripts/roofline_thp_exec.py" "$root/scripts/roofline_thp_study_test.py"
    "$root/scripts/roofline_actions.py" "$root/scripts/roofline_operator.py"
    "$root/scripts/roofline_reuse_diagnostic.py" "$root/scripts/roofline_calibration.py"
    "$root/scripts/roofline_sweep_validation.py" "$root/jobs/roofline_spmv_timing.sh"
    "$root/src/drivers/cvfem_hex8_ns_upwind_bench.cpp" "$root/src/drivers/cvfem_perf_control.hpp"
    "$root/src/tests/verification/cvfem_roofline_window.hpp")
cp "${sources[@]}" "$out/"
sha256sum "${sources[@]}" > "$out/measurement-source.sha256"
set -- --binary "$bench" --out "$out/captures"
if [[ -n $option ]]; then set -- "$@" "$option"; fi
python3 "$root/scripts/roofline_spmv_timing.py" "$@"
echo "SPMV TIMING DIAGNOSTIC COLLECTED; performance campaign remains unvalidated: $out"
