#!/usr/bin/env bash
# Configure/build pysfem in build_py and run the Nitsche Hertz spike.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${ROOT}/build_py"
VENV_PY="${ROOT}/venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "Missing ${VENV_PY}. Create the venv and install python/requirements.txt + nanobind." >&2
    exit 1
fi

mkdir -p "${BUILD}"
if [[ ! -f "${BUILD}/CMakeCache.txt" ]]; then
    cmake -S "${ROOT}" -B "${BUILD}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DSFEM_ENABLE_PYTHON=ON \
        -DSFEM_ENABLE_CUDA=OFF \
        -DPython_EXECUTABLE="${VENV_PY}"
fi

cmake --build "${BUILD}" --target pysfem -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

export PYTHONPATH="${BUILD}/python/bindings:${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
exec "${VENV_PY}" "${ROOT}/python/spike/nitsche_contact.py" --check "$@"
