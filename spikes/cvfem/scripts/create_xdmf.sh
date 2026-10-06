#!/usr/bin/env bash
# Build a ParaView XDMF file from cvfem_hex8_ns_steady output.
#
#   SFEM_CASE=couette SFEM_N=8 ./cvfem_hex8_ns_steady out_dir
#   ./scripts/create_xdmf.sh out_dir
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIKE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${SPIKE_ROOT}/../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

exec "${PYTHON}" "${SPIKE_ROOT}/python/create_xdmf.py" "$@"
