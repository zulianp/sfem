#!/usr/bin/env bash
# Keep all references and operator measurements in the same exclusive allocation.
set -euo pipefail
bench=${1:?usage: roofline_remaining.sh benchmark gemm-binary blas-library /new/results/directory}
gemm=${2:?missing DGEMM binary}
blas=${3:?missing BLAS library}
out=${4:?missing results directory}
[[ $# == 4 && ${SLURM_CPUS_PER_TASK:-0} == 72 ]] || { echo 'Request 72 CPU cores'; exit 1; }
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir "$out"
out=$(cd "$out" && pwd)
exec > >(tee "$out/run.log") 2>&1
date -u
hostname
PYTHONPATH="$root/scripts${PYTHONPATH:+:$PYTHONPATH}" python3 - "$gemm" "$blas" "$out" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
from roofline_gemm import library_identity, parse_output
binary, library, out = map(Path, sys.argv[1:])
env = dict(os.environ, OMP_NUM_THREADS='72', OPENBLAS_NUM_THREADS='72', GOTO_NUM_THREADS='72')
command = [str(binary), 'preflight', '128', '72']
(out/'gemm-preflight-command.json').write_text(json.dumps(command)+'\n')
with (out/'gemm-preflight.out').open('w') as log:
    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
output = (out/'gemm-preflight.out').read_text()
identity = library_identity(output, 72, library, production=True)
parse_output(output, 'normal', 1, n=128, threads=72)
(out/'gemm-library.json').write_text(json.dumps(identity, indent=2)+'\n')
PY
env -u CVFEM_ROOFLINE_THP_POLICY python3 "$root/scripts/roofline_operator_test.py" "$bench"
env -u CVFEM_ROOFLINE_THP_POLICY python3 "$root/scripts/roofline_actions_test.py" "$bench"
env -u CVFEM_ROOFLINE_THP_POLICY python3 "$root/scripts/roofline_gemm_test.py" --binary "$gemm"
if [[ ${CVFEM_ROOFLINE_THP_POLICY:-} == disabled ]]; then
    env -u CVFEM_ROOFLINE_THP_POLICY python3 "$root/scripts/roofline_memory_policy_test.py" "$bench" --gemm "$gemm"
fi
# Calibration failure stops the campaign before any operator is accepted.
bash "$root/jobs/roofline_calibration.sh" "$out/calibration" --validate-reuse
sources=("$root/scripts/roofline_memory_policy.py" "$root/scripts/roofline_thp_exec.py" "$root/scripts/roofline_memory_policy_test.py"
    "$root/scripts/roofline_operator.py" "$root/scripts/roofline_operator_test.py"
    "$root/scripts/roofline_actions.py" "$root/scripts/roofline_actions_test.py"
    "$root/scripts/roofline_gemm.py" "$root/scripts/roofline_gemm_test.py"
    "$root/scripts/roofline_reuse_diagnostic.py" "$root/jobs/roofline_remaining.sh"
    "$root/src/drivers/cvfem_hex8_ns_upwind_bench.cpp" "$root/src/drivers/cvfem_roofline_gemm.cpp"
    "$root/src/drivers/cvfem_perf_control.hpp" "$root/src/drivers/cvfem_calibration_session.hpp"
    "$root/src/tests/verification/cvfem_roofline_window.hpp")
cp "${sources[@]}" "$out/"
sha256sum "${sources[@]}" > "$out/measurement-source.sha256"
python3 "$root/scripts/roofline_operator.py" --binary "$bench" \
    --out "$out/residuals" --calibration "$out/calibration/captures" --long-windows
python3 "$root/scripts/roofline_actions.py" --binary "$bench" \
    --out "$out/actions" --calibration "$out/calibration/captures" --long-windows
python3 "$root/scripts/roofline_gemm.py" --binary "$gemm" --blas-library "$blas" \
    --out "$out/gemm" --calibration "$out/calibration/captures"
echo "MATCHED RESIDUAL, ACTION AND COMPUTE STUDY PASSED: $out"
