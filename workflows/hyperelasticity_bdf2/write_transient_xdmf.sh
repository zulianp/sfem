#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 <mesh> <output-dir> [output.xdmf]" >&2
    exit 1
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd -P)"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
MESH="$1"
OUTPUT_DIR="$2"
XDMF="${3:-$OUTPUT_DIR/output.xdmf}"
FIELDS_DIR="${SFEM_FIELDS_DIR:-$OUTPUT_DIR/out}"
TIME_FILE="$FIELDS_DIR/time.txt"
XDMF_DIR="$(cd -- "$(dirname -- "$XDMF")" >/dev/null 2>&1 && pwd -P)"
XDMF_NAME="$(basename -- "$XDMF")"
XDMF_MESH="$OUTPUT_DIR/.xdmf_mesh"

if [[ ! -f "$TIME_FILE" ]]; then
    echo "missing transient time log: $TIME_FILE" >&2
    exit 1
fi

shopt -s nullglob
patterns=()
for field in disp velocity acceleration gap contact_force; do
    for component in 0 1 2; do
        files=("$FIELDS_DIR/$field.$component".*.*)
        if ((${#files[@]})); then
            patterns+=("$FIELDS_DIR/$field.$component.*.*")
        fi
    done
done
shopt -u nullglob

if ((${#patterns[@]} == 0)); then
    echo "no transient raw fields found in $FIELDS_DIR" >&2
    exit 1
fi

point_data="$(IFS=,; echo "${patterns[*]}")"

"$PYTHON" "$ROOT_DIR/workflows/hyperelasticity_bdf2/prepare_xdmf_mesh.py" "$MESH" "$XDMF_MESH"

if [[ -x "$ROOT_DIR/build64/raw_to_db" ]]; then
    (
        cd "$XDMF_DIR"
        "$ROOT_DIR/build64/raw_to_db" "$XDMF_MESH" "$XDMF_NAME" -p "$point_data" --transient --time_whole_txt="$TIME_FILE"
    )
elif command -v raw_to_db >/dev/null 2>&1; then
    (
        cd "$XDMF_DIR"
        raw_to_db "$XDMF_MESH" "$XDMF_NAME" -p "$point_data" --transient --time_whole_txt="$TIME_FILE"
    )
elif command -v raw_to_db.py >/dev/null 2>&1; then
    (
        cd "$XDMF_DIR"
        raw_to_db.py "$XDMF_MESH" "$XDMF_NAME" -p "$point_data" --transient --time_whole_txt="$TIME_FILE"
    )
else
    (
        cd "$XDMF_DIR"
        PYTHONPATH="$ROOT_DIR/external/smesh/python/smesh:${PYTHONPATH:-}" \
            "$PYTHON" "$ROOT_DIR/external/smesh/python/smesh/raw_to_db.py" \
            "$XDMF_MESH" "$XDMF_NAME" -p "$point_data" --transient --time_whole_txt="$TIME_FILE"
    )
fi

rm -rf "$XDMF_MESH"
echo "Wrote transient XDMF: $XDMF"
