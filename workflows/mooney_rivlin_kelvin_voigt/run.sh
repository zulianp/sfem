#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF2'
usage:
  run.sh [output-dir] [key=value ...]

Runs the elasticity playground on pulse.yaml: a 2 x 1 x 1 Mooney-Rivlin Kelvin-Voigt cantilever,
clamped on its min-x face and struck on its max-x face by a half-sine traction pulse, implicit
Newmark. Every key=value is passed on as --set key=value.

It writes:

  <output-dir>/run.log               the playground's log
  <output-dir>/history.csv           every step: Newton and linear iterations, the tip displacement
  <output-dir>/output.xdmf (+ .h5)   transient disp/velocity/acceleration (ParaView)

Defaults: output-dir workflows/mooney_rivlin_kelvin_voigt/output

Examples:
  run.sh out mesh.box.n=[16,8,8]                       refine
  run.sh out material.eta_s=0.005                      more damping
  run.sh out "loads.neumann[0].value=[-50]"            a harder strike

Useful environment:
  SFEM_BUILD_DIR        build directory, default: build64
  SFEM_SKIP_BUILD       set to 1 to skip building the playground
EOF2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd -P)"
WORKFLOW_DIR="$ROOT_DIR/workflows/mooney_rivlin_kelvin_voigt"
BUILD_DIR="${SFEM_BUILD_DIR:-$ROOT_DIR/build64}"
TARGET=playground
EXE="$BUILD_DIR/$TARGET"

OUTPUT_DIR="${1:-$WORKFLOW_DIR/output}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd -P)"
shift $(($# > 1 ? 1 : $#))
SETS=()
for kv in "$@"; do
    SETS+=(--set "$kv")
done

if [[ "${SFEM_SKIP_BUILD:-0}" != "1" ]]; then
    cmake --build "$BUILD_DIR" --target "$TARGET"
fi

if [[ ! -x "$EXE" ]]; then
    echo "missing executable: $EXE" >&2
    echo "build it with: cmake --build $BUILD_DIR --target $TARGET" >&2
    exit 1
fi

# Field files are named by an export counter: a shorter re-run would leave a longer one's behind.
rm -rf "$OUTPUT_DIR/out" "$OUTPUT_DIR/mesh"

start=$(date +%s)
"$EXE" "$WORKFLOW_DIR/pulse.yaml" --set "output.path=$OUTPUT_DIR" --set "output.history_csv=$OUTPUT_DIR/history.csv" \
    ${SETS[@]+"${SETS[@]}"} > "$OUTPUT_DIR/run.log" 2>&1 || {
    tail -5 "$OUTPUT_DIR/run.log" >&2
    echo "$TARGET failed (log: $OUTPUT_DIR/run.log)" >&2
    exit 1
}
end=$(date +%s)
echo "Ran $TARGET in $((end - start)) s (log: $OUTPUT_DIR/run.log)"

"$ROOT_DIR/workflows/hyperelasticity_bdf2/write_transient_xdmf.sh" \
    "$OUTPUT_DIR/mesh" "$OUTPUT_DIR" "$OUTPUT_DIR/output.xdmf"
echo "XDMF output: $OUTPUT_DIR/output.xdmf"
