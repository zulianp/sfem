#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-"$ROOT/venv/bin/python"}"

cd "$ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

echo "M9 required Python and generated-code regression tests"
"$PYTHON_BIN" -m unittest \
    python.codegen.framework.tests.test_symbolic \
    python.codegen.framework.tests.test_gen_api \
    python.codegen.framework.tests.test_residual \
    python.codegen.framework.tests.test_neohookean_ogden \
    python.codegen.framework.tests.test_m9_regression

if command -v mpic++ >/dev/null 2>&1 || command -v mpicxx >/dev/null 2>&1 || command -v c++ >/dev/null 2>&1; then
    echo "Generated OpenMP compile checks: covered by test_m9_regression and existing unittest gates"
else
    echo "Generated OpenMP compile checks: skipped because no C++ compiler is available"
fi

if find "$ROOT" -name ryml.hpp -print -quit | grep -q .; then
    echo "Generated wrapper syntax checks: covered by test_m9_regression"
else
    echo "Generated wrapper syntax checks: skipped where ryml.hpp is unavailable"
fi

if command -v nvcc >/dev/null 2>&1; then
    echo "Optional CUDA checks: nvcc available; CUDA unittest gates ran above"
else
    echo "Optional CUDA checks: skipped because nvcc is unavailable"
fi

echo "M9 regression entry point completed"
