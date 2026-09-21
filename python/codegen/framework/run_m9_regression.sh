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
    python.codegen.framework.tests.test_assembly_plans_name_the_kernels \
    python.codegen.framework.tests.test_geometry_plan_agrees \
    python.codegen.framework.tests.test_apply_variants \
    python.codegen.framework.tests.test_apply_variants_drive_emission \
    python.codegen.framework.tests.test_apply_variants_match_generation \
    python.codegen.framework.tests.test_residual_path_capabilities \
    python.codegen.framework.tests.test_kernel_ast_slice \
    python.codegen.framework.tests.test_target_binding \
    python.codegen.framework.tests.test_kernel_function_ir \
    python.codegen.framework.tests.test_kernel_ast_passes \
    python.codegen.framework.tests.test_emission_is_a_printer || unittest_status=$?

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

# The manifest gate that stood here, `codegen_snapshot verify`, is retired.  It
# asked whether the generator still agreed with a record of itself, and its
# record went unwritten for sixty-three commits while the tree gate below passed
# on every one of them: dark, and counted as cover.  The question below is the
# one worth asking, and CI runs it.

# This asks whether
# frontend/ops/generated -- the tree CMake compiles into libsfem -- is what the
# generator produces today.  CI runs this one.
if [[ "${SFEM_CODEGEN_TREE:-0}" == "1" ]]; then
    echo "Shipped-tree gate: comparing frontend/ops/generated against a fresh generation"
    "$PYTHON_BIN" -m codegen.framework.tools.codegen_snapshot check-tree --quiet
else
    echo "Shipped-tree gate: skipped; set SFEM_CODEGEN_TREE=1 to run it (~6 min)"
fi

if [[ "${SFEM_REPRODUCIBILITY:-0}" == "1" ]]; then
    echo "Input-output reproducibility gate: every kernel's answers against the baseline"
    "$PYTHON_BIN" -m codegen.framework.tools.reproducibility --all --refine 3
else
    echo "Input-output reproducibility gate: skipped; set SFEM_REPRODUCIBILITY=1 to run it (needs a C++ compiler)"
fi

if [[ "${SFEM_APPLY_BENCH:-0}" == "1" ]]; then
    echo "Matrix-free apply gate: parity and throughput"
    "$PYTHON_BIN" -m codegen.framework.tools.apply_bench --element HEX8 --refine 30
else
    echo "Matrix-free apply gate: skipped; set SFEM_APPLY_BENCH=1 to run it (needs a C++ compiler)"
fi

echo "M9 regression entry point completed"
exit "$unittest_status"
