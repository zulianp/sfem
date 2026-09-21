#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-"$ROOT/venv/bin/python"}"
OUTPUT_DIR="${1:-/private/tmp/sfem_wall_hump_high_re}"

: "${SFEM_REYNOLDS_NUMBER:=2000}"
: "${SFEM_HUMP_INLET_U:=1.0}"
: "${SFEM_HUMP_BODY_LENGTH:=1.0}"
: "${SFEM_HUMP_NX:=16}"
: "${SFEM_HUMP_NY:=6}"
: "${SFEM_HUMP_NZ:=2}"
: "${SFEM_ELEM_TYPE:=HEX27}"
: "${SFEM_DT:=0.01}"
: "${SFEM_MAX_STEPS:=10}"
: "${SFEM_NL_MAX_IT:=20}"
: "${SFEM_LSOLVE_MAX_IT:=8000}"
: "${SFEM_NL_ATOL:=1.0e-7}"
: "${SFEM_LSOLVE_ATOL:=1.0e-10}"
: "${SFEM_NL_ALPHA:=0.5}"

if [[ -z "${SFEM_NU+x}" ]]; then
    SFEM_NU="$(
        "$PYTHON_BIN" -c \
            'import sys; u=float(sys.argv[1]); L=float(sys.argv[2]); re=float(sys.argv[3]); print("{:.17g}".format(u * L / re))' \
            "$SFEM_HUMP_INLET_U" "$SFEM_HUMP_BODY_LENGTH" "$SFEM_REYNOLDS_NUMBER"
    )"
fi

if [[ -z "${OMPI_COMM_WORLD_SIZE+x}" ]]; then
    : "${OMPI_MCA_btl:=self}"
    export OMPI_MCA_btl
fi

export SFEM_REYNOLDS_NUMBER
export SFEM_HUMP_INLET_U
export SFEM_HUMP_BODY_LENGTH
export SFEM_HUMP_NX
export SFEM_HUMP_NY
export SFEM_HUMP_NZ
export SFEM_ELEM_TYPE
export SFEM_DT
export SFEM_MAX_STEPS
export SFEM_NL_MAX_IT
export SFEM_LSOLVE_MAX_IT
export SFEM_NL_ATOL
export SFEM_LSOLVE_ATOL
export SFEM_NL_ALPHA
export SFEM_NU

printf 'wall_mounted_hump high-Re run\n'
printf '  Re=%s, U_ref=%s, L_ref=%s, nu=%s\n' \
    "$SFEM_REYNOLDS_NUMBER" "$SFEM_HUMP_INLET_U" "$SFEM_HUMP_BODY_LENGTH" "$SFEM_NU"
printf '  mesh=%sx%sx%s, element=%s, dt=%s, steps=%s\n' \
    "$SFEM_HUMP_NX" "$SFEM_HUMP_NY" "$SFEM_HUMP_NZ" "$SFEM_ELEM_TYPE" "$SFEM_DT" "$SFEM_MAX_STEPS"

exec "$ROOT/drivers/simulations/run_wall_mounted_hump.sh" "$OUTPUT_DIR"
