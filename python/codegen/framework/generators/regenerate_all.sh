#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." >/dev/null 2>&1 && pwd -P)"
PYTHON="${SFEM_PYTHON:-$ROOT_DIR/venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/python${PYTHONPATH:+:$PYTHONPATH}"

run_generator() {
    local module="$1"
    shift
    printf '\n==> %s %s\n' "$module" "$*"
    "$PYTHON" -m "$module" "$@"
}

run_generator codegen.framework.generators.linear_elasticity
run_generator codegen.framework.generators.laplace
run_generator codegen.framework.generators.neohookean_ogden
run_generator codegen.framework.generators.saint_venant_kirchhoff
run_generator codegen.framework.generators.modified_mooney_rivlin
run_generator codegen.framework.generators.mooney_rivlin_kelvin_voigt_newmark
run_generator codegen.framework.generators.neumann
# shellcheck disable=SC2086
run_generator codegen.framework.generators.neumann_general ${SFEM_NEUMANN_GENERAL_ARGS:-}
run_generator codegen.framework.generators.poro_elasticity
# shellcheck disable=SC2086
run_generator codegen.framework.generators.stokes ${SFEM_STOKES_ARGS:-}
run_generator codegen.framework.generators.two_phase_flow

if [[ "${SFEM_GENERATE_CUDA:-0}" == "1" ]]; then
    run_generator codegen.framework.generators.cuda ${SFEM_CUDA_ARGS:-}
fi

if [[ -n "${SFEM_GENERATOR_MANIFESTS:-}" ]]; then
    # shellcheck disable=SC2086
    run_generator codegen.framework.generators.op_registration ${SFEM_GENERATOR_MANIFESTS} ${SFEM_OP_REGISTRATION_ARGS:-}
fi
