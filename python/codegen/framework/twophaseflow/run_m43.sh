#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${SFEM_BUILD_DIR:-${ROOT_DIR}/build}"
RESULT_DIR="${SFEM_M43_RESULT_DIR:-${BUILD_DIR}/two_phase_flow_m43}"

source "${ROOT_DIR}/venv/bin/activate"

PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}" \
python -m codegen.framework.materials.two_phase_flow \
    --out-dir "${ROOT_DIR}/python/codegen/framework/twophaseflow/generated"
cmake --build "${BUILD_DIR}" --target generated_two_phase_flow -j "${SFEM_BUILD_JOBS:-4}"
python "${ROOT_DIR}/python/codegen/framework/twophaseflow/verify_two_phase_flow.py" \
    --driver "${BUILD_DIR}/generated_two_phase_flow" \
    --work-dir "${RESULT_DIR}/runs" \
    --summary "${RESULT_DIR}/summary.md" \
    --benchmark-repeats "${SFEM_BENCHMARK_REPEATS:-100}"
