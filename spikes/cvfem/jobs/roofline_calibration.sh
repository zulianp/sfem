#!/usr/bin/env bash
# Run inside an exclusive one-socket Grace allocation, with the compiler uenv active.
set -euo pipefail
out=${1:?usage: roofline_calibration.sh /new/results/directory}
mode=${2:-}
[[ $# -le 2 && ( -z "$mode" || "$mode" == --diagnose-writes || "$mode" == --diagnose-phases || "$mode" == --validate-sweeps || "$mode" == --diagnose-warmup || "$mode" == --diagnose-reuse || "$mode" == --validate-reuse ) ]] || { echo 'Unknown calibration option'; exit 1; }
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir "$out"
out=$(cd "$out" && pwd)
exec > >(tee "$out/run.log") 2>&1
date -u
hostname
printf 'RESULTS=%s\n' "$out"
[[ ${SLURM_CPUS_PER_TASK:-0} == 72 ]] || { echo 'Request 72 CPU cores'; exit 1; }
sources=("$root/src/drivers/cvfem_perf_control.hpp" "$root/src/drivers/cvfem_calibration_session.hpp" "$root/src/drivers/cvfem_roofline_calibration.cpp" "$root/scripts/roofline_calibration.py" "$root/jobs/roofline_calibration.sh")
if [[ "$mode" == --diagnose-phases || "$mode" == --validate-sweeps || "$mode" == --diagnose-warmup || "$mode" == --diagnose-reuse || "$mode" == --validate-reuse ]]; then
    sources+=("$root/scripts/roofline_phase_control_test.py")
fi
if [[ "$mode" == --validate-sweeps || "$mode" == --diagnose-warmup || "$mode" == --diagnose-reuse || "$mode" == --validate-reuse ]]; then
    sources+=("$root/scripts/roofline_sweep_validation.py" "$root/scripts/roofline_sweep_validation_test.py")
fi
if [[ "$mode" == --diagnose-warmup || "$mode" == --diagnose-reuse || "$mode" == --validate-reuse ]]; then
    sources+=("$root/scripts/roofline_warmup_diagnostic.py")
fi
if [[ "$mode" == --diagnose-reuse || "$mode" == --validate-reuse ]]; then
    sources+=("$root/scripts/roofline_reuse_diagnostic.py" "$root/scripts/roofline_reuse_diagnostic_test.py")
fi
sha256sum "${sources[@]}" > "$out/source.sha256"
cp "${sources[@]}" "$out/"
"${CXX:-g++}" --version > "$out/compiler.txt"
python3 -c 'import sys; print(sys.executable); print(sys.version)' > "$out/python-version.txt"
flags=(-O3 -DNDEBUG -fopenmp -std=c++17 -mcpu=native -ffp-contract=fast)
build_command=("${CXX:-g++}" "${flags[@]}" "$root/src/drivers/cvfem_roofline_calibration.cpp" -o "$out/calibration")
printf '%q ' "${build_command[@]}" > "$out/build-command.txt"
printf '\n' >> "$out/build-command.txt"
"${build_command[@]}"
python3 "$root/scripts/roofline_calibration.py" --selftest
if [[ "$mode" == --diagnose-phases || "$mode" == --validate-sweeps || "$mode" == --diagnose-warmup || "$mode" == --diagnose-reuse || "$mode" == --validate-reuse ]]; then
    python3 "$root/scripts/roofline_phase_control_test.py" "$out/calibration"
fi
collector_command=(python3 "$root/scripts/roofline_calibration.py" --binary "$out/calibration" --out "$out/captures")
if [[ "$mode" == --validate-sweeps ]]; then
    python3 "$root/scripts/roofline_sweep_validation_test.py"
    collector_command=(python3 "$root/scripts/roofline_sweep_validation.py" --binary "$out/calibration" --out "$out/captures")
elif [[ "$mode" == --diagnose-reuse || "$mode" == --validate-reuse ]]; then
    python3 "$root/scripts/roofline_reuse_diagnostic_test.py" "$out/calibration"
    collector_command=(python3 "$root/scripts/roofline_reuse_diagnostic.py" --binary "$out/calibration" --out "$out/captures")
    if [[ "$mode" == --validate-reuse ]]; then collector_command+=(--validate-reuse); fi
elif [[ "$mode" == --diagnose-warmup ]]; then
    python3 "$root/scripts/roofline_warmup_diagnostic.py" --selftest
    collector_command=(python3 "$root/scripts/roofline_warmup_diagnostic.py" --binary "$out/calibration" --out "$out/captures")
elif [[ -n "$mode" ]]; then
    collector_command+=("$mode")
fi
"${collector_command[@]}"
if [[ "$mode" == --validate-sweeps ]]; then
    echo "ROOFLINE SWEEP COUNTER VALIDATION PASSED: $out"
elif [[ "$mode" == --validate-reuse ]]; then
    echo "ROOFLINE REUSE CONTROL VALIDATION PASSED: $out"
elif [[ "$mode" == --diagnose-reuse ]]; then
    echo "ROOFLINE REUSE DIAGNOSTIC COLLECTED: $out"
elif [[ "$mode" == --diagnose-warmup ]]; then
    echo "ROOFLINE WARMUP DIAGNOSTIC COLLECTED: $out"
elif [[ "$mode" == --diagnose-phases ]]; then
    echo "ROOFLINE PHASE DIAGNOSTIC COLLECTED: $out"
elif [[ "$mode" == --diagnose-writes ]]; then
    echo "ROOFLINE WRITE DIAGNOSTIC COLLECTED: $out"
else
    echo "ROOFLINE COUNTER GATE PASSED: $out"
fi
