#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
usage:
  run.sh [cantilever|contact] [output-dir] [key=value ...]

Runs the elasticity playground on cantilever.yaml or contact.yaml (Prony-series Mooney-Rivlin,
implicit Newmark), then visualises it. Every key=value is passed on as --set key=value.

  cantilever  a 10 x 1 x 1 box, HEX8, clamped on the min-x face, sagging under gravity in -y
  contact     a free 2 x 1 x 1 block, no Dirichlet conditions, falling onto a hemisphere on
              a floor: frictionless contact of its whole surface with the obstacle's signed
              distance, contact-stabilised Newmark

It writes:

  <output-dir>/run.log               the playground's log
  <output-dir>/history.csv           every step: Newton and linear iterations, and the probes
  <output-dir>/output.xdmf (+ .h5)   transient disp/velocity/acceleration (ParaView); with
                                     contact also gap and contact_force
  <output-dir>/sdf/sdf.xdmf          contact: the obstacle's signed distance; contour it at 0
  <output-dir>/response.csv          the response per exported step
  <output-dir>/newton.csv            Newton iterations and final measure per step
  <output-dir>/gravity_response.png  the response and the Newton iterations over time
  <output-dir>/deformed.png          the last frame, deformed (only when pvbatch is found)

The run stops if a step does not converge.

Defaults: case cantilever, output-dir workflows/mooney_rivlin_gravity/output/<case>

Examples:
  run.sh contact out mesh.box.n=[24,24,24]          refine the block
  run.sh contact out obstacle.shape.type=slope      plane, slope, staircase or hemisphere
  run.sh cantilever out time.t_end=1

Useful environment:
  SFEM_BUILD_DIR        build directory, default: build64
  SFEM_SKIP_BUILD       set to 1 to skip building the playground
  SFEM_PVBATCH          pvbatch executable, default: found on PATH or in /Applications
  SFEM_SKIP_RENDER      set to 1 to skip the ParaView render
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CASE="${1:-cantilever}"
case "$CASE" in
    cantilever | contact) ;;
    *)
        echo "unknown case: $CASE (cantilever or contact)" >&2
        exit 1
        ;;
esac

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd -P)"
WORKFLOW_DIR="$ROOT_DIR/workflows/mooney_rivlin_gravity"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
BUILD_DIR="${SFEM_BUILD_DIR:-$ROOT_DIR/build64}"
TARGET=playground
EXE="$BUILD_DIR/$TARGET"

OUTPUT_DIR="${2:-$WORKFLOW_DIR/output/$CASE}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd -P)"
shift $(($# > 2 ? 2 : $#))
SETS=()
for kv in "$@"; do
    SETS+=(--set "$kv")
done

if [[ ! -x "$PYTHON" ]]; then
    echo "missing Python interpreter: $PYTHON" >&2
    exit 1
fi

if [[ "${SFEM_SKIP_BUILD:-0}" != "1" ]]; then
    cmake --build "$BUILD_DIR" --target "$TARGET"
fi

if [[ ! -x "$EXE" ]]; then
    echo "missing executable: $EXE" >&2
    echo "build it with: cmake --build $BUILD_DIR --target $TARGET" >&2
    exit 1
fi

# Field files are named by an export counter: a shorter re-run would leave a longer one's behind.
rm -rf "$OUTPUT_DIR/out" "$OUTPUT_DIR/mesh" "$OUTPUT_DIR/sdf"
CASE_FILE="$WORKFLOW_DIR/$CASE.yaml"

start=$(date +%s)
"$EXE" "$CASE_FILE" --set "output.path=$OUTPUT_DIR" --set "output.history_csv=$OUTPUT_DIR/history.csv" ${SETS[@]+"${SETS[@]}"} \
    > "$OUTPUT_DIR/run.log" 2>&1 || {
    tail -5 "$OUTPUT_DIR/run.log" >&2
    echo "$TARGET $CASE failed (log: $OUTPUT_DIR/run.log)" >&2
    exit 1
}
end=$(date +%s)
echo "Ran $TARGET $CASE in $((end - start)) s (log: $OUTPUT_DIR/run.log)"

"$ROOT_DIR/workflows/hyperelasticity_bdf2/write_transient_xdmf.sh" \
    "$OUTPUT_DIR/mesh" "$OUTPUT_DIR" "$OUTPUT_DIR/output.xdmf"

"$PYTHON" "$WORKFLOW_DIR/plot_response.py" "$CASE" "$OUTPUT_DIR" "$CASE_FILE" "$@"

if [[ "${SFEM_SKIP_RENDER:-0}" != "1" ]]; then
    PVBATCH="${SFEM_PVBATCH:-$(command -v pvbatch || true)}"
    if [[ -z "$PVBATCH" ]]; then
        for app in /Applications/ParaView*.app; do
            if [[ -x "$app/Contents/bin/pvbatch" ]]; then
                PVBATCH="$app/Contents/bin/pvbatch"
            fi
        done
    fi
    if [[ -n "$PVBATCH" ]]; then
        "$PVBATCH" "$WORKFLOW_DIR/render_deformed.py" "$CASE" "$OUTPUT_DIR/output.xdmf" "$OUTPUT_DIR/deformed.png" \
            "$OUTPUT_DIR/sdf/sdf.xdmf"
    else
        echo "pvbatch not found: skipping deformed.png (set SFEM_PVBATCH)"
    fi
fi
