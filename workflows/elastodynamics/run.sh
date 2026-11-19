#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF2'
usage:
  run.sh [output-dir] [key=value ...]

Runs the elasticity playground on case.yaml: a free linear elastic block, falling under gravity
onto a hemisphere on a floor; frictionless contact of its whole surface, implicit Newmark. The
mesh is a level-4 semi-structured refinement and SSMGC solves the obstacle. Every key=value is
passed on as --set key=value.

It writes:

  <output-dir>/run.log               the playground's log
  <output-dir>/history.csv           every step: Newton and SSMGC iterations, contact nodes, reaction
  <output-dir>/output.xdmf (+ .h5)   transient disp/velocity/acceleration/gap/contact_force (ParaView)
  <output-dir>/sdf/sdf.xdmf          the obstacle's signed distance; contour it at 0

The run stops if a step does not converge.

Defaults: output-dir workflows/elastodynamics/output

Examples:
  run.sh out mesh.refine=8                             a finer level (8 x 4 x 4 macro elements of 8^3)
  run.sh out obstacle.backend=node_sdf                 the nodal-SDF obstacle (MPRGP) instead
  run.sh out execution_space=device                    on the GPU
  run.sh out obstacle.ssmgc.nlsmooth_steps=10          an SSMGC key over the playground's

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
WORKFLOW_DIR="$ROOT_DIR/workflows/elastodynamics"
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
rm -rf "$OUTPUT_DIR/out" "$OUTPUT_DIR/mesh" "$OUTPUT_DIR/sdf"

start=$(date +%s)
"$EXE" "$WORKFLOW_DIR/case.yaml" --set "output.path=$OUTPUT_DIR" --set "output.history_csv=$OUTPUT_DIR/history.csv" \
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
