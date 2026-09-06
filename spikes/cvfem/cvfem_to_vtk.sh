#!/usr/bin/env bash
# Convert a cvfem_hex8_ns_ssgmg output folder to a .vtu for ParaView.
#
#   ./cvfem_to_vtk.sh <output_folder> [output.vtu]
#
# The driver writes mesh/ plus one file per field: vel.0 vel.1 vel.2 (velocity) and p
# (pressure). x.* is the older naming and is still picked up if present. On a semi-structured run mesh/ is the *expanded* fine mesh, written by
# semistructured_export_as_standard, so the fields line up with it node for node;
# coarse_mesh/ holds the macro-element mesh and is not what you want to view.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd -P)"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
RAW_TO_DB="$ROOT/external/smesh/python/smesh/raw_to_db.py"

OUT="${1:?usage: cvfem_to_vtk.sh <output_folder> [output.vtu]}"
VTU="${2:-$OUT/solution.vtu}"

[[ -x "$PYTHON"    ]] || { echo "missing python: $PYTHON" >&2; exit 1; }
[[ -f "$RAW_TO_DB" ]] || { echo "missing raw_to_db: $RAW_TO_DB" >&2; exit 1; }
[[ -d "$OUT/mesh"  ]] || { echo "missing $OUT/mesh -- was SFEM_ENABLE_OUTPUT=0?" >&2; exit 1; }

shopt -s nullglob
fields=("$OUT"/vel.*.float64 "$OUT"/vel.*.float32 "$OUT"/p.float64 "$OUT"/p.float32 \
        "$OUT"/x.*.float64 "$OUT"/x.*.float32)
shopt -u nullglob
[[ ${#fields[@]} -gt 0 ]] || { echo "no vel.*/p field files in $OUT" >&2; exit 1; }

PYTHONPATH="$ROOT/external/smesh/python/smesh${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$RAW_TO_DB" "$OUT/mesh" "$VTU" -p "$(IFS=,; echo "${fields[*]}")"

echo "wrote $VTU"
