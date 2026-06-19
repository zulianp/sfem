#!/usr/bin/env bash

set -euo pipefail

SCRIPTPATH="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPTPATH}/.." >/dev/null 2>&1 ; pwd -P)"
REMOTE="${SFEM_SYNC_REMOTE:-alps:/capstor/scratch/cscs/zulianp/sfem/}"

CUBQL_DIR="${REPO_ROOT}/external/ssdf/external/cubql"
for required_file in \
    "cuBQL/builder/cpu.h" \
    "cuBQL/builder/cpu/instantiate_builders.cpp"
do
    if [[ ! -f "${CUBQL_DIR}/${required_file}" ]]; then
        echo "Missing ${CUBQL_DIR}/${required_file}" >&2
        echo "Run: git -C ${REPO_ROOT}/external/ssdf submodule update --init --recursive --force external/cubql" >&2
        exit 1
    fi
done

RSYNC_EXCLUDES=(
    --exclude 'api'
    --exclude 'venv'
    --exclude '/build*/'
    --exclude '*.o'
    --exclude '.DS_Store'
    --exclude '.git/'
    --exclude '.git'
    --exclude 'benchmark/db'
    --exclude '.vscode'
    --exclude '.venv'
)

set -x

# --delete 
rsync -av "${RSYNC_EXCLUDES[@]}" "${REPO_ROOT}/" "${REMOTE}"
