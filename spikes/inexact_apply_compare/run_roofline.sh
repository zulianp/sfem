#!/usr/bin/env bash
# Roofline model of the inexact-apply kernels, from the generated source.
#
#   run_roofline.sh [material] [element] [machine]
#
# element is TET4 (default), HEX8 or TET10; machine is grace (default) or m1max.
# Reuses the tree `run_split.sh` generates, so running the benchmark first costs
# nothing here and running this first costs nothing there.
#
# Writes a text report and an SVG beside the log.  The measured points come from
# `measured.json`, which records what was actually run and on what; a kernel with
# no measurement is still modelled, it just has no dot on the plot.
set -euo pipefail

MATERIAL="${1:-neohookean_ogden}"
ELEMENT="${2:-TET4}"
MACHINE="${3:-grace}"
LOWER="$(echo "$ELEMENT" | tr '[:upper:]' '[:lower:]')"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../.." && pwd)"
SFEM="${SFEM_MAIN_CHECKOUT:-$WORKTREE/../sfem}"
PYTHON="${SFEM_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for candidate in "$SFEM/.venv/bin/python" "$SFEM/venv/bin/python" \
                     "$WORKTREE/.venv/bin/python" "$WORKTREE/venv/bin/python"; do
        [ -x "$candidate" ] && PYTHON="$candidate" && break
    done
fi
: "${PYTHON:?set SFEM_PYTHON: no venv found beside the checkout}"
BUILD="${SFEM_BUILD:-$SFEM/build}"
WORK="${SFEM_SPIKE_WORK:-${TMPDIR:-/tmp}}/inexact_apply_split"
OUT="${SFEM_ROOFLINE_OUT:-$WORK}"
LOG="$OUT/roofline_${MATERIAL}_${LOWER}_${MACHINE}.log"
# One measurement file per machine: a dof rate from a laptop and one from a
# Grace socket are not comparable, and a single file would invite mixing them.
MEASURED="$HERE/measured_${MACHINE}.json"
[ -f "$MEASURED" ] || MEASURED="$HERE/measured_grace.json"

mkdir -p "$WORK" "$OUT"
echo "START $(date +%T)  material=$MATERIAL element=$ELEMENT machine=$MACHINE" | tee "$LOG"

GEN="$WORK/gen/$MATERIAL/d3/$LOWER"
if [ ! -f "$GEN/${MATERIAL}_${LOWER}_inexact_apply_inline.hpp" ]; then
    echo "[1/2] generating kernels (slow for HEX8)" | tee -a "$LOG"
    ( cd "$WORKTREE" && PYTHONPATH=python:python/codegen/framework/materials \
        "$PYTHON" - "$WORK/gen" "$MATERIAL" "$ELEMENT" <<'PY'
import dataclasses, sys
out, name, element = sys.argv[1], sys.argv[2], sys.argv[3]
from sfem import gen
module = __import__(name)
gen.generate(dataclasses.replace(module.material, inexact_apply=True),
             "%s/%s" % (out, name), elements=(element,), clean=False)
PY
    ) 2>&1 | stdbuf -oL tee -a "$LOG"
else
    echo "[1/2] kernels already generated, reusing" | tee -a "$LOG"
fi

# The mesh decides how much of a gather is compulsory, so the model is given the
# same mesh the measurements were taken on rather than a default of one.
NPE="$("$PYTHON" - "$MEASURED" "$ELEMENT" "$MACHINE" <<'PY'
import json, sys
measured = json.load(open(sys.argv[1]))
for point in measured.get("points", []):
    if point.get("element") == sys.argv[2] and measured.get("machine") == sys.argv[3]:
        print("%.6f" % (point["nnodes"] / float(point["nelements"])))
        break
else:
    print("1.0")
PY
)"
THREADS="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("threads") or "")' "$MEASURED")"
echo "[2/2] modelling (nodes per element $NPE, threads ${THREADS:-all})" | tee -a "$LOG"

CONFIGS=()
for candidate in "$BUILD/external/smesh/smesh_config.hpp" "$BUILD/external/smesh/sfem_config.h" \
                 "$WORKTREE/build/external/smesh/smesh_config.hpp" \
                 "$WORKTREE/configuration/sfem_config.h.in"; do
    [ -f "$candidate" ] && CONFIGS+=(--config "$candidate")
done

cd "$WORKTREE/python"
PYTHONPATH=. "$PYTHON" -m codegen.framework.tools.roofline "$WORK/gen" \
    --kernel "${MATERIAL}_${LOWER}_inexact_apply" \
    --machine "$MACHINE" \
    ${THREADS:+--threads "$THREADS"} \
    --nodes-per-element "$NPE" \
    --measured "$MEASURED" \
    --plot "$OUT/roofline_${MATERIAL}_${LOWER}_${MACHINE}.pdf" \
    "${CONFIGS[@]}" 2>&1 | stdbuf -oL tee -a "$LOG"

echo "END $(date +%T)  report $LOG" | tee -a "$LOG"
