#!/usr/bin/env bash
# Run any case in cases/ end to end: mesh, solve, ParaView export, analysis.
#
#   ./run_case.sh <case-name> [workdir]
#   ./run_case.sh                       # lists the cases
#
# The mesh is built once per workdir and reused, so running several cases into the same workdir
# only pays for it the first time. The driver is invoked from the workdir because SFEM resolves
# the paths inside an inline dirichlet block against the working directory.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd -- "$HERE/../.." >/dev/null 2>&1 && pwd -P)"
PYTHON="${PRONY_PYTHON:-$ROOT_DIR/venv/bin/python}"
EXE="${PRONY_EXE:-$HERE/build/prony_visco_torsion}"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <case-name> [workdir]" >&2
    echo >&2
    echo "available cases:" >&2
    for f in "$HERE"/cases/*.yaml; do
        printf '  %-28s %s\n' "$(basename "$f" .yaml)" \
            "$(grep -m1 '^# ' "$f" | sed 's/^# //')" >&2
    done
    exit 1
fi

CASE_NAME="$(basename "$1" .yaml)"
CASE_FILE="$HERE/cases/$CASE_NAME.yaml"
WORKDIR="${2:-$HERE/output}"

[[ -f "$CASE_FILE" ]] || { echo "no such case: $CASE_FILE" >&2; exit 1; }
[[ -x "$EXE" ]] || { echo "missing executable: $EXE -- build the spike first" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "missing python: $PYTHON (set PRONY_PYTHON)" >&2; exit 1; }

mkdir -p "$WORKDIR"

# The viscoelastic kernels are HEX8 only. Built once per workdir.
if [[ ! -d "$WORKDIR/mesh" ]]; then
    echo ">>> building mesh in $WORKDIR/mesh"
    "$PYTHON" "$ROOT_DIR/python/sfem/mesh/box_mesh.py" "$WORKDIR/mesh" \
        --cell_type=HEX8 \
        -x "${PRONY_NX:-16}" -y "${PRONY_NY:-5}" -z "${PRONY_NZ:-5}" \
        --width "${PRONY_LENGTH:-1.0}" --height "${PRONY_HEIGHT:-0.2}" --depth "${PRONY_DEPTH:-0.2}"
fi

cp "$CASE_FILE" "$WORKDIR/"

# Single-threaded by default: at these problem sizes the constraint handling inside every CG
# iteration costs more to hand to an OpenMP team than to run. See the README.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

RESULTS="$(sed -n 's/^  path: *//p' "$CASE_FILE" | head -1)"
RESULTS="${RESULTS:-output}"

# Clear this case's results before running, and only this case's -- the mesh beside it is
# deliberately reused. Field files are named by an export counter, not by time, so a shorter
# re-run leaves the previous run's higher-index files behind and the transient export then sees
# more field files than there are time steps. That failure surfaces far downstream, in raw_to_db,
# as an "Invalid sequence length" on whichever field happens to be inconsistent.
rm -rf "${WORKDIR:?}/$RESULTS"

cd "$WORKDIR"
PRONY_EXE="$EXE" PRONYRUN_DIR="$WORKDIR/logs" \
    "$HERE/tools/pronyrun.sh" "$CASE_NAME" "$CASE_NAME.yaml"

if [[ -f "$WORKDIR/$RESULTS/out/time.txt" ]]; then
    "$ROOT_DIR/workflows/hyperelasticity_bdf2/write_transient_xdmf.sh" \
        "$WORKDIR/$RESULTS/mesh" "$WORKDIR/$RESULTS" "$WORKDIR/$RESULTS/output.xdmf" \
        || echo "[warn] XDMF export failed; the raw fields are still in $RESULTS/out" >&2
fi

echo
if grep -q '^  profile: cyclic' "$CASE_NAME.yaml"; then
    "$PYTHON" "$HERE/tools/analyse_hysteresis.py" "$WORKDIR/$RESULTS/history.csv" --case "$WORKDIR/$CASE_NAME.yaml"
else
    "$PYTHON" "$HERE/tools/validate_torsion_release.py" "$WORKDIR/$RESULTS/history.csv" --case "$WORKDIR/$CASE_NAME.yaml"
fi

echo
echo "$CASE_NAME:"
echo "  history:  $WORKDIR/$RESULTS/history.csv"
echo "  paraview: $WORKDIR/$RESULTS/output.xdmf"
echo "  raw log:  $WORKDIR/logs/$CASE_NAME.log"
