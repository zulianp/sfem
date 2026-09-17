#!/usr/bin/env bash
#
# Regenerate every material.
#
# The materials are independent: each writes its own subtree, and the only
# files they share are the primitive headers, which they all write with
# identical content through an atomic rename.  So they run concurrently, which
# is the whole of the speed-up -- generation is single-threaded inside a
# material, so a sequential run left most of the machine idle for its whole
# duration.
#
#   SFEM_GENERATOR_JOBS   how many materials at once (default: the core count)
#   SFEM_PYTHON           the interpreter to use
#   SFEM_GENERATOR_LOGS   where the per-material logs go
#
# Each material gets its own log rather than interleaving into one stream, and
# a failure prints that log, so a broken run says which material broke and
# why rather than leaving a shuffled transcript to untangle.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." >/dev/null 2>&1 && pwd -P)"
PYTHON="${SFEM_PYTHON:-$ROOT_DIR/venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/python${PYTHONPATH:+:$PYTHONPATH}"

# sympy caches its expression constructors, and the default ceiling of 1000
# entries is far below what a hyperelastic material needs: the caches sit full
# and evict entries that are about to be asked for again.  Raising it is worth
# about a fifth of the wall time on mooney_rivlin_kelvin_voigt_newmark, for
# byte-identical output, and it saturates well below this value.
export SYMPY_CACHE_SIZE="${SYMPY_CACHE_SIZE:-100000}"

if [[ -n "${SFEM_GENERATOR_JOBS:-}" ]]; then
    JOBS="$SFEM_GENERATOR_JOBS"
elif command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
elif command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
else
    JOBS=4
fi

LOG_DIR="${SFEM_GENERATOR_LOGS:-${TMPDIR:-/tmp}/sfem_regenerate_all.$$}"
mkdir -p "$LOG_DIR"

declare -a PIDS=()
declare -a NAMES=()
declare -a LOGS=()

start_generator() {
    local module="$1"
    shift
    local label="${module##*.}"
    local log="$LOG_DIR/$label.log"
    printf '==> %s starting (%s)\n' "$label" "$(date +%T)"
    ( "$PYTHON" -m "$module" "$@" ) >"$log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$label")
    LOGS+=("$log")
}

# Wait until fewer than JOBS are running, so a long material does not hold the
# rest back and the machine is not oversubscribed either.
await_slot() {
    while (( $(jobs -rp | wc -l) >= JOBS )); do
        wait -n 2>/dev/null || true
    done
}

run_generator() {
    await_slot
    start_generator "$@"
}

# Wait for the wave that is running, report each material by name, and stop the
# script if any of them failed.  A function rather than a block because the
# device wave below drains the same way, and a second copy of this loop would
# be a second place for the reporting to drift.
drain_generators() {
    local status=0 index
    for index in "${!PIDS[@]}"; do
        if wait "${PIDS[$index]}"; then
            printf '==> %s done (%s)\n' "${NAMES[$index]}" "$(date +%T)"
        else
            status=1
            printf '==> %s FAILED (%s); its log follows\n' "${NAMES[$index]}" "$(date +%T)"
            cat "${LOGS[$index]}"
        fi
    done
    PIDS=(); NAMES=(); LOGS=()
    if (( status != 0 )); then
        printf 'logs: %s\n' "$LOG_DIR"
        exit "$status"
    fi
}

# One list, used for both targets.  `generators/cuda.py` used to carry its own
# and they had drifted apart -- it named three materials the host tree does not
# carry and missed four that it does -- so the device tree could not be the
# host tree's twin however it was invoked.  Every material generator already
# takes `--target`, so the device pass is these same generators with the target
# switched rather than a second generator.
generate_materials() {
    run_generator codegen.framework.generators.linear_elasticity "$@"
    run_generator codegen.framework.generators.laplace "$@"
    run_generator codegen.framework.generators.neohookean_ogden "$@"
    run_generator codegen.framework.generators.mooney_rivlin_kelvin_voigt_newmark "$@"
    run_generator codegen.framework.generators.neumann "$@"
    # shellcheck disable=SC2086
    run_generator codegen.framework.generators.neumann_general ${SFEM_NEUMANN_GENERAL_ARGS:-} "$@"
    run_generator codegen.framework.generators.two_phase_flow "$@"
    run_generator codegen.framework.generators.navier_stokes "$@"
}

generate_materials
drain_generators

# The headers that belong to a target rather than to a material.  The OpenMP set
# falls out of the material runs above; the CUDA set does not, because CUDA
# generation is opt-in and also writes an untracked per-material operator tree.
# Asking the backend for them needs no material, so it runs every time and the
# `.cuh` files stop being four tracked files that nothing regenerates.
printf '==> shared headers\n'
# `set -e` would abort on a bare call, but without naming the step that failed,
# and this one matters: `codegen_snapshot check-tree` expects these files now,
# so a failure that is not read as a failure here surfaces later as tree drift
# somewhere else entirely.  The materials above report themselves by name; so
# does this.
if ! "$PYTHON" -m codegen.framework.generators.shared_headers; then
    printf '==> shared headers FAILED\n'
    exit 1
fi

# The device tree: the same materials, the same generators, `--target cuda`.
# It lands in `cuda/` folders inside each material's own tree, beside the host
# sources it mirrors, which is where the rest of the repository keeps its
# device sources.  After the host wave rather than beside it, because the two
# share a directory and both write the target-independent matrix-format files.
if [[ "${SFEM_GENERATE_CUDA:-0}" == "1" ]]; then
    printf '==> cuda\n'
    # shellcheck disable=SC2086
    generate_materials --target cuda ${SFEM_CUDA_ARGS:-}
    drain_generators
fi

if [[ -n "${SFEM_GENERATOR_MANIFESTS:-}" ]]; then
    printf '==> op_registration\n'
    # shellcheck disable=SC2086
    "$PYTHON" -m codegen.framework.generators.op_registration \
        ${SFEM_GENERATOR_MANIFESTS} ${SFEM_OP_REGISTRATION_ARGS:-}
fi

printf 'all materials regenerated; logs: %s\n' "$LOG_DIR"
