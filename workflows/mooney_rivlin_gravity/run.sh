#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
usage:
  run.sh [cantilever|contact] [output-dir]

Runs sfem_MooneyRivlinGravityTest with the Prony-series Mooney-Rivlin material and implicit
Newmark, then visualises it.

  cantilever  a 10 x 1 x 1 box, HEX8, clamped on the min-x face, sagging under gravity in -y
              (the test test_mooney_rivlin_gravity)
  contact     a free 2 x 1 x 1 block, no Dirichlet conditions, falling onto a hemisphere on
              a floor (the test test_mooney_rivlin_gravity_contact): frictionless contact of
              its whole surface with the obstacle's exact signed distance, contact-stabilised
              Newmark

It writes:

  <output-dir>/run.log               the test's log
  <output-dir>/output.xdmf (+ .h5)   transient disp/velocity/acceleration (ParaView); with
                                     contact also gap and contact_force
  <output-dir>/response.csv          the response per exported step
  <output-dir>/newton.csv            Newton iterations and final KKT residual per step
  <output-dir>/gravity_response.png  the response and the Newton iterations over time
  <output-dir>/deformed.png          the last frame, deformed (only when pvbatch is found)

The test fails if a step does not converge; the run then stops.

Defaults: case cantilever, output-dir workflows/mooney_rivlin_gravity/output/<case>

Useful environment:
  SFEM_BUILD_DIR        build directory, default: build64
  SFEM_SKIP_BUILD       set to 1 to skip building the test
  SFEM_PVBATCH          pvbatch executable, default: found on PATH or in /Applications
  SFEM_SKIP_RENDER      set to 1 to skip the ParaView render
  SFEM_BASE_RESOLUTION  cantilever: elements across the section, the length gets 10x (4);
                        contact: elements along each edge of the block (8)
  SFEM_DENSITY          cantilever 1, contact 10
  SFEM_GRAVITY          cantilever 0.5, contact 9.81
  SFEM_C10, SFEM_C01    cantilever 800.622, 800.108; contact 4000, 2000
  SFEM_BULK_MODULUS     cantilever 4000, contact 160000
  SFEM_PRONY_G          cantilever 0.4,0.4,0.1,0.05; contact the test's default
  SFEM_PRONY_TAU        cantilever 1,2,5,10; contact the test's default
  SFEM_USE_WLF          default: 0
  SFEM_DT               cantilever 0.005, contact 0.01
  SFEM_T                cantilever 7.5, contact 2
  SFEM_EXPORT_FREQ      default: 5
  SFEM_OBSTACLE_TYPE    contact: 0 plane, 1 sloping plane, 2 staircase, 4 hemisphere (4);
                        the test lists the parameters of each
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CASE="${1:-cantilever}"
case "$CASE" in
    cantilever) TEST=test_mooney_rivlin_gravity ;;
    contact) TEST=test_mooney_rivlin_gravity_contact ;;
    *)
        echo "unknown case: $CASE (cantilever or contact)" >&2
        exit 1
        ;;
esac

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd -P)"
WORKFLOW_DIR="$ROOT_DIR/workflows/mooney_rivlin_gravity"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
BUILD_DIR="${SFEM_BUILD_DIR:-$ROOT_DIR/build64}"
TARGET=sfem_MooneyRivlinGravityTest
EXE="$BUILD_DIR/$TARGET"

OUTPUT_DIR="${2:-$WORKFLOW_DIR/output/$CASE}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd -P)"
# The test writes its fields to a folder named after it, relative to its working directory.
FIELDS_DIR="$OUTPUT_DIR/$TEST"
MESH_DIR="$OUTPUT_DIR/mesh"

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

export SFEM_USE_WLF="${SFEM_USE_WLF:-0}"
export SFEM_EXPORT_FREQ="${SFEM_EXPORT_FREQ:-5}"
export SFEM_ENABLE_OUTPUT=1

rm -rf "$MESH_DIR" "$FIELDS_DIR"
if [[ "$CASE" == cantilever ]]; then
    export SFEM_BASE_RESOLUTION="${SFEM_BASE_RESOLUTION:-4}"
    export SFEM_DENSITY="${SFEM_DENSITY:-1}"
    export SFEM_GRAVITY="${SFEM_GRAVITY:-0.5}"
    export SFEM_C10="${SFEM_C10:-800.622}"
    export SFEM_C01="${SFEM_C01:-800.108}"
    export SFEM_BULK_MODULUS="${SFEM_BULK_MODULUS:-4000}"
    export SFEM_PRONY_G="${SFEM_PRONY_G:-0.4,0.4,0.1,0.05}"
    export SFEM_PRONY_TAU="${SFEM_PRONY_TAU:-1,2,5,10}"
    export SFEM_DT="${SFEM_DT:-0.005}"
    export SFEM_T="${SFEM_T:-7.5}"
    # Clamp the min-x face, gravity along -y.
    export SFEM_CONTACT_DIR=0
    export SFEM_FIX_SIDE=0
    export SFEM_GRAVITY_DIR=1
    export SFEM_MESH="$MESH_DIR"
    "$PYTHON" "$ROOT_DIR/python/sfem/mesh/box_mesh.py" "$MESH_DIR" \
        -c hex8 \
        -x "$((10 * SFEM_BASE_RESOLUTION + 1))" \
        -y "$((SFEM_BASE_RESOLUTION + 1))" \
        -z "$((SFEM_BASE_RESOLUTION + 1))" \
        --width=10 \
        --height=1 \
        --depth=1
else
    # The test's own block, a gap above the obstacle; gravity and contact along -z.
    export SFEM_BASE_RESOLUTION="${SFEM_BASE_RESOLUTION:-8}"
    export SFEM_DENSITY="${SFEM_DENSITY:-10}"
    export SFEM_GRAVITY="${SFEM_GRAVITY:-9.81}"
    # Stiff enough that the hemisphere does not punch into the block on a fine mesh: with
    # C10 = 50 the bump inverts elements at 24 elements per edge.
    export SFEM_C10="${SFEM_C10:-4000}"
    export SFEM_C01="${SFEM_C01:-2000}"
    export SFEM_BULK_MODULUS="${SFEM_BULK_MODULUS:-160000}"
    export SFEM_DT="${SFEM_DT:-0.01}"
    export SFEM_T="${SFEM_T:-2}"
fi

start=$(date +%s)
(cd "$OUTPUT_DIR" && "$EXE" "$TEST") > "$OUTPUT_DIR/run.log" 2>&1 || {
    tail -5 "$OUTPUT_DIR/run.log" >&2
    echo "$TARGET $TEST failed (log: $OUTPUT_DIR/run.log)" >&2
    exit 1
}
end=$(date +%s)
echo "Ran $TARGET $TEST in $((end - start)) s (log: $OUTPUT_DIR/run.log)"

SFEM_FIELDS_DIR="$FIELDS_DIR" "$ROOT_DIR/workflows/hyperelasticity_bdf2/write_transient_xdmf.sh" \
    "$FIELDS_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR/output.xdmf"

"$PYTHON" "$WORKFLOW_DIR/plot_response.py" "$CASE" "$OUTPUT_DIR" "$FIELDS_DIR"

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
            "$FIELDS_DIR/sdf/sdf.xdmf"
    else
        echo "pvbatch not found: skipping deformed.png (set SFEM_PVBATCH)"
    fi
fi
