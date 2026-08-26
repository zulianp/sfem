#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
usage:
  run_and_export.sh [mesh|AUTO] [dirichlet.yaml|NONE] [neumann.yaml|NONE] [output-dir]

Defaults:
  mesh        AUTO, creates a released 3D brick
  dirichlet  AUTO, fixed left face when mesh is AUTO
  neumann    AUTO, zero right-face traction when mesh is AUTO
  output-dir workflows/mooney_rivlin_kelvin_voigt_newmark/output

Useful environment:
  SFEM_BUILD_DIR       build directory, default: build64
  SFEM_SKIP_BUILD      set to 1 to skip building the driver
  SFEM_RAW_TO_DB       explicit raw_to_db executable/script
  SFEM_MU             default: 1000
  SFEM_LAMBDA         default: 1000
  SFEM_ETA_S          default: 0.00005
  SFEM_ETA_B          default: 0
  SFEM_RHO            default: 0.01
  SFEM_DT             default: 0.001
  SFEM_STEPS          default: 4000
  SFEM_EXPORT_FREQ    default: 10
  SFEM_LOAD_SCALE     default: 0
  SFEM_LOAD_PULSE_TIME default: 0, set >0 for a half-sine pulse
  SFEM_INITIAL_DISP_Y default: -0.4
  SFEM_INITIAL_DISP_Z default: 0.15
  SFEM_XDMF           default: <output-dir>/output.xdmf
  SFEM_AUTO_ELEM_TYPE default: HEX8, supported: HEX8 or HEX27
  SFEM_AUTO_NX        default: 8
  SFEM_AUTO_NY        default: 4
  SFEM_AUTO_NZ        default: 4
  SFEM_AUTO_LENGTH    default: 2
  SFEM_AUTO_HEIGHT    default: 1
  SFEM_AUTO_WIDTH     default: 1
  SFEM_AUTO_TRACTION  default: [0.0, 0.0, 0.0]

Examples:
  workflows/mooney_rivlin_kelvin_voigt_newmark/run_and_export.sh

  SFEM_STEPS=20 SFEM_DT=0.005 \
    workflows/mooney_rivlin_kelvin_voigt_newmark/run_and_export.sh \
    workflows/hyperelasticity/geometry_hex8/box NONE NONE /tmp/mr_kv_box
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd -P)"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
BUILD_DIR="${SFEM_BUILD_DIR:-$ROOT_DIR/build64}"
EXE="${SFEM_EXE:-$BUILD_DIR/mooney_rivlin_kelvin_voigt_newmark}"

MESH="${1:-${SFEM_MESH:-AUTO}}"
DIRICHLET="${2:-${SFEM_DIRICHLET:-NONE}}"
NEUMANN="${3:-${SFEM_NEUMANN:-NONE}}"
OUTPUT_DIR="${4:-${SFEM_OUTPUT_DIR:-$ROOT_DIR/workflows/mooney_rivlin_kelvin_voigt_newmark/output}}"
XDMF="${SFEM_XDMF:-$OUTPUT_DIR/output.xdmf}"

if [[ ! -x "$PYTHON" ]]; then
    echo "missing Python interpreter: $PYTHON" >&2
    exit 1
fi

if [[ "${SFEM_SKIP_BUILD:-0}" != "1" ]]; then
    cmake --build "$BUILD_DIR" --target mooney_rivlin_kelvin_voigt_newmark
fi

if [[ ! -x "$EXE" ]]; then
    echo "missing executable: $EXE" >&2
    echo "build it with: cmake --build $BUILD_DIR --target mooney_rivlin_kelvin_voigt_newmark" >&2
    exit 1
fi

rm -rf "$OUTPUT_DIR/out" "$OUTPUT_DIR/mesh"
mkdir -p "$OUTPUT_DIR" "$(dirname -- "$XDMF")"

if [[ "$MESH" == "AUTO" || "$MESH" == "auto" || -z "$MESH" ]]; then
    auto_elem_type="${SFEM_AUTO_ELEM_TYPE:-${SFEM_ELEM_TYPE:-HEX8}}"
    MESH="$OUTPUT_DIR/input_${auto_elem_type}_bar"
    rm -rf "$MESH"
    "$PYTHON" - "$MESH" <<'PY'
from pathlib import Path
import os
import sys
import numpy as np

mesh = Path(sys.argv[1])
mesh.mkdir(parents=True, exist_ok=True)

nx = int(os.environ.get("SFEM_AUTO_NX", 8))
ny = int(os.environ.get("SFEM_AUTO_NY", 4))
nz = int(os.environ.get("SFEM_AUTO_NZ", 4))
length = float(os.environ.get("SFEM_AUTO_LENGTH", 2.0))
height = float(os.environ.get("SFEM_AUTO_HEIGHT", 1.0))
width = float(os.environ.get("SFEM_AUTO_WIDTH", 1.0))
elem_type = os.environ.get("SFEM_AUTO_ELEM_TYPE", os.environ.get("SFEM_ELEM_TYPE", "HEX8")).upper()

if nx < 1 or ny < 1 or nz < 1:
    raise SystemExit("SFEM_AUTO_NX, SFEM_AUTO_NY, and SFEM_AUTO_NZ must be positive")

if elem_type not in ("HEX8", "HEX27"):
    raise SystemExit(f"unsupported SFEM_AUTO_ELEM_TYPE={elem_type}; use HEX8 or HEX27")

if elem_type == "HEX27":
    gx, gy, gz = 2 * nx + 1, 2 * ny + 1, 2 * nz + 1
    elem_num_nodes = 27
    sfem_hex27_to_cartesian = (
        0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23,
        25, 21, 9, 11, 17, 15, 10, 14, 16, 12, 4, 22, 13,
    )
else:
    gx, gy, gz = nx + 1, ny + 1, nz + 1
    elem_num_nodes = 8

points = []
for iz in range(gz):
    z = width * iz / (gz - 1)
    for iy in range(gy):
        y = height * iy / (gy - 1)
        for ix in range(gx):
            x = length * ix / (gx - 1)
            points.append((x, y, z))
points = np.asarray(points, dtype=np.float32)

def node(ix, iy, iz):
    return ix + gx * (iy + gy * iz)

connectivity = [[] for _ in range(elem_num_nodes)]
left_parent = []
right_parent = []
e = 0
for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            if elem_type == "HEX27":
                cartesian = []
                for lz in range(3):
                    for ly in range(3):
                        for lx in range(3):
                            cartesian.append(node(2 * ix + lx, 2 * iy + ly, 2 * iz + lz))
                nodes = tuple(cartesian[k] for k in sfem_hex27_to_cartesian)
            else:
                nodes = (
                    node(ix, iy, iz),
                    node(ix + 1, iy, iz),
                    node(ix + 1, iy + 1, iz),
                    node(ix, iy + 1, iz),
                    node(ix, iy, iz + 1),
                    node(ix + 1, iy, iz + 1),
                    node(ix + 1, iy + 1, iz + 1),
                    node(ix, iy + 1, iz + 1),
                )
            for k, n in enumerate(nodes):
                connectivity[k].append(n)
            if ix == 0:
                left_parent.append(e)
            if ix == nx - 1:
                right_parent.append(e)
            e += 1

elements_yaml = "".join(f"- i{i}: i{i}.raw\n" for i in range(elem_num_nodes))
mesh.joinpath("meta.yaml").write_text(f"""# SFEM mesh meta file
spatial_dimension: 3
elem_num_nodes: {elem_num_nodes}
element_type: {elem_type}
n_elements: {e}
n_nodes: {len(points)}
elements:
{elements_yaml}
points:
- x: x.raw
- y: y.raw
- z: z.raw
rpath: true
""")
points[:, 0].tofile(mesh / "x.raw")
points[:, 1].tofile(mesh / "y.raw")
points[:, 2].tofile(mesh / "z.raw")
for i, data in enumerate(connectivity):
    np.asarray(data, dtype=np.int32).tofile(mesh / f"i{i}.raw")

def write_sideset(path, parent, lfi):
    path.mkdir(parents=True, exist_ok=True)
    np.asarray(parent, dtype=np.int32).tofile(path / "parent.raw")
    np.full(len(parent), lfi, dtype=np.int16).tofile(path / "lfi.int16.raw")
    path.joinpath("meta.yaml").write_text(f"""# SFEM sideset meta file
size: {len(parent)}
parent: parent.raw
lfi: lfi.int16.raw
rpath: true
""")

# HEX8 and HEX27 use the same hexahedron local face ids:
# x-max/right is lfi=1, x-min/left is lfi=3.
write_sideset(mesh / "sidesets" / "left", left_parent, 3)
write_sideset(mesh / "sidesets" / "right", right_parent, 1)
PY

    DIRICHLET="$OUTPUT_DIR/fixed_left.yaml"
    NEUMANN="$OUTPUT_DIR/right_face_load.yaml"
    cat > "$DIRICHLET" <<YAML
dirichlet_conditions:
  - type: sideset
    format: file
    path: $MESH/sidesets/left
    value: [0, 0, 0]
    component: [0, 1, 2]
YAML

    auto_traction_x="${SFEM_AUTO_TRACTION_X:-0.0}"
    auto_traction_y="${SFEM_AUTO_TRACTION_Y:-0.0}"
    auto_traction_z="${SFEM_AUTO_TRACTION_Z:-0.0}"
    if "$PYTHON" - "$auto_traction_x" "$auto_traction_y" "$auto_traction_z" <<'PY'
import sys
raise SystemExit(0 if all(abs(float(v)) == 0.0 for v in sys.argv[1:]) else 1)
PY
    then
        NEUMANN="NONE"
    else
        cat > "$NEUMANN" <<YAML
neumann_conditions:
  - type: sideset
    format: file
    path: $MESH/sidesets/right
    value: [$auto_traction_x, $auto_traction_y, $auto_traction_z]
    component: [0, 1, 2]
YAML
    fi
fi

export SFEM_MU="${SFEM_MU:-1000}"
export SFEM_LAMBDA="${SFEM_LAMBDA:-1000}"
export SFEM_ETA_S="${SFEM_ETA_S:-0.00005}"
export SFEM_ETA_B="${SFEM_ETA_B:-0}"
export SFEM_RHO="${SFEM_RHO:-0.01}"
export SFEM_DT="${SFEM_DT:-0.001}"
export SFEM_STEPS="${SFEM_STEPS:-4000}"
export SFEM_EXPORT_FREQ="${SFEM_EXPORT_FREQ:-10}"
export SFEM_VERBOSE="${SFEM_VERBOSE:-0}"
export SFEM_LOAD_RAMP_TIME="${SFEM_LOAD_RAMP_TIME:-0}"
export SFEM_LOAD_PULSE_TIME="${SFEM_LOAD_PULSE_TIME:-0}"
export SFEM_LOAD_SCALE="${SFEM_LOAD_SCALE:-0}"
export SFEM_INITIAL_DISP_Y="${SFEM_INITIAL_DISP_Y:--0.4}"
export SFEM_INITIAL_DISP_Z="${SFEM_INITIAL_DISP_Z:-0.15}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-self}"

"$EXE" "$MESH" "$DIRICHLET" "$NEUMANN" "$OUTPUT_DIR"

run_raw_to_db() {
    if [[ -n "${SFEM_RAW_TO_DB:-}" ]]; then
        "$SFEM_RAW_TO_DB" "$@"
    elif command -v raw_to_db >/dev/null 2>&1; then
        command raw_to_db "$@"
    elif command -v raw_to_db.py >/dev/null 2>&1; then
        command raw_to_db.py "$@"
    else
        PYTHONPATH="$ROOT_DIR/external/smesh/python/smesh:${PYTHONPATH:-}" \
            "$PYTHON" "$ROOT_DIR/external/smesh/python/smesh/raw_to_db.py" "$@"
    fi
}

collect_component() {
    local field="$1"
    local component="$2"
    find "$OUTPUT_DIR/out" -maxdepth 1 -type f -name "$field.$component.*.*" | sort
}

disp0=($(collect_component disp 0))
disp1=($(collect_component disp 1))
disp2=($(collect_component disp 2))
vel0=($(collect_component velocity 0))
vel1=($(collect_component velocity 1))
vel2=($(collect_component velocity 2))
acc0=($(collect_component acceleration 0))
acc1=($(collect_component acceleration 1))
acc2=($(collect_component acceleration 2))

nsteps=${#disp0[@]}
for count in "${#disp1[@]}" "${#disp2[@]}" "${#vel0[@]}" "${#vel1[@]}" "${#vel2[@]}" "${#acc0[@]}" "${#acc1[@]}" "${#acc2[@]}"; do
    if (( count < nsteps )); then
        nsteps=$count
    fi
done

if (( nsteps == 0 )); then
    echo "no complete displacement/velocity/acceleration outputs found in $OUTPUT_DIR/out" >&2
    exit 1
fi

VECTOR_DIR="$OUTPUT_DIR/out/xdmf_vectors"
rm -rf "$VECTOR_DIR"
mkdir -p "$VECTOR_DIR"
"$PYTHON" - "$OUTPUT_DIR/out" "$VECTOR_DIR" "$nsteps" <<'PY'
from pathlib import Path
import sys
import numpy as np

source = Path(sys.argv[1])
target = Path(sys.argv[2])
nsteps = int(sys.argv[3])

for field in ("disp", "velocity", "acceleration"):
    components = [sorted(source.glob(f"{field}.{c}.*.*")) for c in range(3)]
    for files in components:
        if len(files) < nsteps:
            raise SystemExit(f"not enough {field} component files for vector export")

    for step in range(nsteps):
        data = [np.fromfile(components[c][step], dtype=np.float64) for c in range(3)]
        if data[0].size != data[1].size or data[0].size != data[2].size:
            raise SystemExit(f"inconsistent {field} component sizes at step {step}")
        np.column_stack(data).tofile(target / f"{field}.vec3.{step:09d}.float64")
PY

transient_point_data="$VECTOR_DIR/disp.vec3.*.float64,$VECTOR_DIR/velocity.vec3.*.float64,$VECTOR_DIR/acceleration.vec3.*.float64"
xdmf_dir="$(cd -- "$(dirname -- "$XDMF")" >/dev/null 2>&1 && pwd -P)"
xdmf_name="$(basename -- "$XDMF")"
(
    cd "$xdmf_dir"
    run_raw_to_db "$OUTPUT_DIR/mesh" "$xdmf_name" -p "$transient_point_data" --transient --time_whole_txt="$OUTPUT_DIR/out/time.txt"
)

echo "Run output: $OUTPUT_DIR"
echo "XDMF output: $XDMF"
