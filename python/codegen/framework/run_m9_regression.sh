#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-"$ROOT/venv/bin/python"}"

cd "$ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

echo "M9 required Python and generated-code regression tests"
# Do not let `set -e` abort here: this suite has known pre-existing failures, and
# the checks below still need to run.  The combined status is returned at the end.
unittest_status=0
"$PYTHON_BIN" -m unittest \
    python.codegen.framework.tests.test_symbolic \
    python.codegen.framework.tests.test_gen_api \
    python.codegen.framework.tests.test_residual \
    python.codegen.framework.tests.test_neohookean_ogden \
    python.codegen.framework.tests.test_m9_regression \
    python.codegen.framework.tests.test_layering \
    python.codegen.framework.tests.test_module_imports \
    python.codegen.framework.tests.test_form_collection_boundary \
    python.codegen.framework.tests.test_emitters_do_not_analyse \
    python.codegen.framework.tests.test_plans_are_consumed \
    python.codegen.framework.tests.test_geometry_plan_agrees \
    python.codegen.framework.tests.test_apply_variants \
    python.codegen.framework.tests.test_apply_variants_drive_emission \
    python.codegen.framework.tests.test_apply_variants_match_generation \
    python.codegen.framework.tests.test_residual_path_capabilities || unittest_status=$?

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

if [[ "${SFEM_CODEGEN_SNAPSHOT:-0}" == "1" ]]; then
    echo "Generated-source snapshot gate: verifying against the committed manifest"
    "$PYTHON_BIN" -m codegen.framework.tools.codegen_snapshot verify --quiet
else
    echo "Generated-source snapshot gate: skipped; set SFEM_CODEGEN_SNAPSHOT=1 to run it (~3 min)"
fi

if [[ "${SFEM_APPLY_BENCH:-0}" == "1" ]]; then
    echo "Matrix-free apply gate: parity and throughput"
    "$PYTHON_BIN" -m codegen.framework.tools.apply_bench --element HEX8 --refine 30
else
    echo "Matrix-free apply gate: skipped; set SFEM_APPLY_BENCH=1 to run it (needs a C++ compiler)"
fi

echo "M9 regression entry point completed"
exit "$unittest_status"
