#!/usr/bin/env bash
# run_transient.sh <case folder> [--sim1-only|--sim2-only] [--set key=value ...]: the valves
# through pump strokes, in <case>/transient/: simulation 1 (released into the cavity, relaxed, at
# rest under the outlet pressure), its last state mapped onto the whole mesh in the file's
# numbering, then simulation 2 (the strokes) from it. The --set overrides go to simulation 2.
# Uses the case scripts' env.sh (SFEM_SOURCE, SFEM_BUILD, PYTHON); LAUNCH prefixes the runs.
set -euo pipefail
CASE="$(cd "$1" && pwd)"; shift
STAGES="sim1 sim2"
case "${1:-}" in
    --sim1-only) STAGES="sim1"; shift ;;
    --sim2-only) STAGES="sim2"; shift ;;
esac
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$CASE/../env.sh"
NAME="$(basename "$CASE")"
[ -f "$HERE/$NAME/sim1.yaml" ] || { echo "run_transient.sh: no $HERE/$NAME/sim1.yaml" >&2; exit 1; }

WORK="$CASE/transient"
mkdir -p "$WORK"
cp "$HERE/$NAME/sim1.yaml" "$HERE/$NAME/sim2.yaml" "$HERE/fp70_membrane_pressure.csv" "$WORK/"
cd "$WORK"

# The Exodus files for ParaView need netCDF4; without it the raw fields are all there is.
EXODUS=1
"$PYTHON" -c "import netCDF4" 2> /dev/null || { EXODUS=0; echo "run_transient.sh: no netCDF4 in $PYTHON, no .e files"; }

for stage in $STAGES; do
    rm -rf "$stage"; mkdir -p "$stage"
    extra=()
    [ "$stage" = sim2 ] && extra=("$@")
    ${LAUNCH:-} "$PLAYGROUND" "$stage.yaml" --set output.path="$stage" --set output.history_csv="$stage/history.csv" \
        ${extra[@]+"${extra[@]}"} 2>&1 | tee "$stage/run.log"
    if [ "$stage" = sim1 ]; then
        # Simulation 2's start, on the whole mesh in the file's numbering, and for ParaView.
        "$PYTHON" "$SMESH_PY/raw_point_data_to_parent.py" sim1/out sim1/mesh "$CASE/db/solid" sim1/whole --fields=disp
        "$PYTHON" "$SMESH_PY/raw_point_data_to_parent.py" sim1/out sim1/mesh "$CASE/db/solid" sim1/view --all-steps
        [ $EXODUS = 0 ] || "$PYTHON" "$SMESH_PY/raw_to_exodusII.py" sim1/view sim1/released.e
    else
        # The strokes for ParaView: every exported step of every field the case writes, on the
        # whole mesh, with its time.
        "$PYTHON" "$SMESH_PY/raw_point_data_to_parent.py" sim2/out sim2/mesh "$CASE/db/solid" sim2/view --all-steps
        [ $EXODUS = 0 ] || "$PYTHON" "$SMESH_PY/raw_to_exodusII.py" sim2/view sim2/strokes.e
    fi
done
echo "run_transient.sh: $WORK"
