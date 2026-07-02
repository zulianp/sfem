#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../" >/dev/null 2>&1 && pwd -P)"
BUILD_DIR="${SFEM_BUILD_DIR:-$ROOT_DIR/build64}"
VENV_PYTHON="${SFEM_PYTHON:-$ROOT_DIR/venv/bin/python}"
WORK_DIR="${SFEM_MLIR_BENCH_DIR:-$ROOT_DIR/IR/bench_tet4_cube}"

CUBE_NX="${SFEM_CUBE_NX:-80}"
CUBE_NY="${SFEM_CUBE_NY:-$CUBE_NX}"
CUBE_NZ="${SFEM_CUBE_NZ:-$CUBE_NX}"
REPEAT="${SFEM_REPEAT:-20}"
WARMUP="${SFEM_WARMUP:-10}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MLIR_OPT_STRATEGY="${SFEM_MLIR_OPT_STRATEGY:-parity}"
OPENCL_MLIR_OPT_STRATEGY="${SFEM_OPENCL_MLIR_OPT_STRATEGY:-$MLIR_OPT_STRATEGY}"
OPENCL_OPT_STRATEGY="${SFEM_OPENCL_OPT_STRATEGY:-parity}"

case "$OPENCL_OPT_STRATEGY" in
    parity|none|baseline)
        OPENCL_BUILD_OPTIONS="-cl-std=CL1.2"
        ;;
    canonical|mad)
        OPENCL_BUILD_OPTIONS="-cl-std=CL1.2 -cl-mad-enable"
        ;;
    scalar|no-signed-zeros)
        OPENCL_BUILD_OPTIONS="-cl-std=CL1.2 -cl-mad-enable -cl-no-signed-zeros"
        ;;
    aggressive|fast-relaxed-math)
        OPENCL_BUILD_OPTIONS="-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable"
        ;;
    *)
        printf 'Unknown SFEM_OPENCL_OPT_STRATEGY=%s (expected parity, canonical, scalar, aggressive)\n' "$OPENCL_OPT_STRATEGY" >&2
        exit 2
        ;;
esac

SMESH_CUBE="${SMESH_CUBE:-$BUILD_DIR/external/smesh/cube}"
GENERATED_TET4_OBJECT="${SFEM_GENERATED_TET4_OBJECT:-$BUILD_DIR/CMakeFiles/sfem.dir/frontend/ops/generated/linear_elasticity/d3/tet4/linear_elasticity_tet4_operator.cpp.o}"
CXX="${CXX:-/opt/homebrew/opt/llvm/bin/clang++}"
if [[ ! -x "$CXX" ]]; then
    CXX="${CXX_FALLBACK:-c++}"
fi
CC="${CC:-/opt/homebrew/opt/llvm/bin/clang}"
if [[ ! -x "$CC" ]]; then
    CC="${CC_FALLBACK:-cc}"
fi

require_file() {
    if [[ ! -e "$1" ]]; then
        printf 'Missing required file: %s\n' "$1" >&2
        exit 1
    fi
}

require_file "$SMESH_CUBE"
require_file "$VENV_PYTHON"
require_file "$GENERATED_TET4_OBJECT"

mkdir -p "$WORK_DIR"
MESH_DIR="$WORK_DIR/mesh"
rm -rf "$MESH_DIR"
mkdir -p "$MESH_DIR"

printf 'Generating TET4 cube: %s x %s x %s -> %s\n' "$CUBE_NX" "$CUBE_NY" "$CUBE_NZ" "$MESH_DIR"
"$SMESH_CUBE" TET4 "$CUBE_NX" "$CUBE_NY" "$CUBE_NZ" 0 0 0 1 1 1 "$MESH_DIR"

MESH_ENV="$WORK_DIR/mesh.env"
"$VENV_PYTHON" - "$MESH_DIR" "$MESH_ENV" <<'PY'
from pathlib import Path
import sys
import numpy as np

mesh = Path(sys.argv[1])
env = Path(sys.argv[2])
i0 = np.fromfile(mesh / "i0.int32", dtype=np.int32)
i1 = np.fromfile(mesh / "i1.int32", dtype=np.int32)
i2 = np.fromfile(mesh / "i2.int32", dtype=np.int32)
i3 = np.fromfile(mesh / "i3.int32", dtype=np.int32)
x = np.fromfile(mesh / "x.float32", dtype=np.float32)
conn = np.stack([i0, i1, i2, i3], axis=1)
degree = np.zeros(x.shape[0], dtype=np.int64)
for elem in conn:
    for node in elem:
        degree[int(node)] += 1
env.write_text(
    "NELEMENTS=%d\nNNODES=%d\nMAX_NODE_DEGREE=%d\n"
    % (conn.shape[0], x.shape[0], int(degree.max(initial=1)))
)
PY
source "$MESH_ENV"

printf 'Mesh: elements=%s nodes=%s max_node_degree=%s\n' "$NELEMENTS" "$NNODES" "$MAX_NODE_DEGREE"

MLIR_C="$WORK_DIR/linear_elasticity_tet4_mlir_apply_openmp_c.c"
MLIR_EMITC="$WORK_DIR/linear_elasticity_tet4_mlir_apply_openmp_c.emitc.mlir"
MLIR_OPENMP_DIR="$WORK_DIR/openmp"
MLIR_OPENCL_DIR="$WORK_DIR/opencl"
MLIR_OPENMP_LL="$MLIR_OPENMP_DIR/linear_elasticity_tet4_mlir_apply_openmp.kernel.ll"
OPENCL_KERNEL="$WORK_DIR/linear_elasticity_tet4_mlir_apply_opencl.cl"
PYTHONPATH="$ROOT_DIR/python${PYTHONPATH:+:$PYTHONPATH}" "$VENV_PYTHON" - "$NELEMENTS" "$NNODES" "$MAX_NODE_DEGREE" "$MLIR_C" "$MLIR_EMITC" "$MLIR_OPENMP_DIR" "$MLIR_OPENMP_LL" "$MLIR_OPENCL_DIR" "$MLIR_OPT_STRATEGY" "$OPENCL_MLIR_OPT_STRATEGY" "$OPENCL_KERNEL" <<'PY'
from pathlib import Path
import sys
from codegen.framework.mlir import MatrixFreeEBEMLIRLowering, _translate_mlir_to_llvm_ir

nelements = int(sys.argv[1])
nnodes = int(sys.argv[2])
max_node_degree = int(sys.argv[3])
c_path = Path(sys.argv[4])
emitc_path = Path(sys.argv[5])
openmp_dir = Path(sys.argv[6])
openmp_ll_path = Path(sys.argv[7])
opencl_dir = Path(sys.argv[8])
optimization_strategy = sys.argv[9]
opencl_optimization_strategy = sys.argv[10]
opencl_kernel_path = Path(sys.argv[11])

root = MatrixFreeEBEMLIRLowering.from_linear_elasticity(element="TET4", vector_size=8)
lowering = root.openmp(
    max_elements=nelements,
    max_nodes=nnodes,
    max_node_degree=max_node_degree,
    optimization_strategy=optimization_strategy,
)
c_path.write_text(lowering.lower_to_c_source())
emitc_path.write_text(lowering.lower_to_emitc_module())
openmp_dir.mkdir(parents=True, exist_ok=True)
(openmp_dir / "linear_elasticity_tet4_mlir_apply_openmp.kernel.scf.mlir").write_text(lowering.render_kernel_scf_module())
(openmp_dir / f"linear_elasticity_tet4_mlir_apply_openmp.kernel.{lowering.optimization_strategy.value}.optimized.scf.mlir").write_text(lowering.optimize_kernel_scf_module())
(openmp_dir / "linear_elasticity_tet4_mlir_apply_openmp.kernel.openmp.mlir").write_text(lowering.lower_kernel_to_openmp_module())
openmp_llvm_mlir = openmp_dir / "linear_elasticity_tet4_mlir_apply_openmp.kernel.llvm.mlir"
openmp_llvm_mlir.write_text(lowering.lower_kernel_to_llvm_module())
_translate_mlir_to_llvm_ir(openmp_llvm_mlir, openmp_ll_path)
opencl = root.opencl(
    max_elements=nelements,
    max_nodes=nnodes,
    max_node_degree=max_node_degree,
    optimization_strategy=opencl_optimization_strategy,
)
opencl.write_inspection_artifacts(opencl_dir)
opencl_kernel_path.write_text(opencl.lower_to_opencl_c_source())
print("mlir_optimization_strategy", lowering.optimization_strategy.value)
print("mlir_optimization_passes", " ".join(lowering.optimization_plan.pre_lowering_passes) or "none")
print("opencl_mlir_optimization_strategy", opencl.optimization_strategy.value)
print("opencl_mlir_optimization_passes", " ".join(opencl.optimization_plan.pre_lowering_passes) or "none")
PY

HARNESS_CPP="$WORK_DIR/bench_mlir_vs_generated_tet4.cpp"
cp "$ROOT_DIR/python/codegen/framework/mlir/cpp/bench_mlir_vs_generated_tet4.cpp" "$HARNESS_CPP"

BENCH_EXE="$WORK_DIR/bench_mlir_vs_generated_tet4"
MLIR_OBJECT="$WORK_DIR/linear_elasticity_tet4_mlir_apply_openmp_c.o"
MLIR_OPENMP_OBJECT="$WORK_DIR/linear_elasticity_tet4_mlir_apply_openmp_kernel.o"
OMP_LIB_DIR="${OMP_LIB_DIR:-/opt/homebrew/opt/llvm/lib}"
if [[ ! -e "$OMP_LIB_DIR/libomp.dylib" ]]; then
    OMP_LIB_DIR="/opt/homebrew/opt/libomp/lib"
fi

"$CC" -O3 -DNDEBUG -std=c11 -c "$MLIR_C" -o "$MLIR_OBJECT"
"$CC" -O3 -DNDEBUG -c "$MLIR_OPENMP_LL" -o "$MLIR_OPENMP_OBJECT"
"$CXX" -O3 -DNDEBUG -std=c++17 \
    "$HARNESS_CPP" "$MLIR_OBJECT" "$MLIR_OPENMP_OBJECT" "$GENERATED_TET4_OBJECT" \
    -L"$OMP_LIB_DIR" -Wl,-rpath,"$OMP_LIB_DIR" -lomp \
    -o "$BENCH_EXE"

export OMP_NUM_THREADS
printf 'Running benchmark: OMP_NUM_THREADS=%s repeat=%s\n' "$OMP_NUM_THREADS" "$REPEAT"
RESULTS_TXT="$WORK_DIR/benchmark_output.txt"
PLOTS_DIR="$WORK_DIR/plots"
mkdir -p "$PLOTS_DIR"
"$BENCH_EXE" "$MESH_DIR" "$NELEMENTS" "$NNODES" "$MAX_NODE_DEGREE" "$REPEAT" | tee "$RESULTS_TXT"

OPENCL_DRIVER_CPP="$WORK_DIR/bench_mlir_opencl_tet4.cpp"
OPENCL_EXE="$WORK_DIR/bench_mlir_opencl_tet4"
cp "$ROOT_DIR/python/codegen/framework/mlir/cpp/bench_mlir_opencl_tet4.cpp" "$OPENCL_DRIVER_CPP"

OPENCL_OUTPUT="$WORK_DIR/opencl_benchmark_output.txt"
OPENCL_TIME="$WORK_DIR/opencl_executable_time.txt"
GENERATED_MELEM="$("$VENV_PYTHON" - "$RESULTS_TXT" <<'PY'
from pathlib import Path
import sys
for line in Path(sys.argv[1]).read_text().splitlines():
    fields = line.split()
    if fields and fields[0] == "generated_apply":
        print(fields[2])
        break
PY
)"
if "$CXX" -O3 -DNDEBUG -std=c++17 "$OPENCL_DRIVER_CPP" "$GENERATED_TET4_OBJECT" \
    -L"$OMP_LIB_DIR" -Wl,-rpath,"$OMP_LIB_DIR" -lomp -framework OpenCL -o "$OPENCL_EXE"; then
    printf 'opencl_runtime_strategy %s\n' "$OPENCL_OPT_STRATEGY" | tee -a "$RESULTS_TXT"
    printf 'opencl_runtime_build_options %s\n' "$OPENCL_BUILD_OPTIONS" | tee -a "$RESULTS_TXT"
    if /usr/bin/time -p "$OPENCL_EXE" "$MESH_DIR" "$OPENCL_KERNEL" "$NELEMENTS" "$NNODES" "$MAX_NODE_DEGREE" "$REPEAT" "$GENERATED_MELEM" "$OPENCL_BUILD_OPTIONS" > "$OPENCL_OUTPUT" 2> "$OPENCL_TIME"; then
        cat "$OPENCL_OUTPUT"
        cat "$OPENCL_OUTPUT" >> "$RESULTS_TXT"
        printf 'opencl_executable_time_file %s\n' "$OPENCL_TIME" | tee -a "$RESULTS_TXT"
        cat "$OPENCL_TIME" | tee -a "$RESULTS_TXT"
    else
        cat "$OPENCL_OUTPUT"
        cat "$OPENCL_TIME"
        printf 'OpenCL benchmark failed; see %s and %s\n' "$OPENCL_OUTPUT" "$OPENCL_TIME" | tee -a "$RESULTS_TXT"
    fi
else
    printf 'OpenCL driver compilation failed\n' | tee -a "$RESULTS_TXT"
fi

MPLCONFIGDIR="$WORK_DIR/matplotlib" "$VENV_PYTHON" - "$RESULTS_TXT" "$PLOTS_DIR" <<'PY'
from pathlib import Path
import csv
import sys

results = Path(sys.argv[1])
plots = Path(sys.argv[2])
plots.mkdir(parents=True, exist_ok=True)

rows = []
for line in results.read_text().splitlines():
    fields = line.split()
    if fields and fields[0] in {"mlir_emitc_apply", "mlir_openmp_apply", "mlir_opencl_apply", "generated_apply"}:
        rows.append(
            {
                "kernel": fields[0],
                "time_s": float(fields[1]),
                "melem_per_s": float(fields[2]),
                "mdof_per_s": float(fields[3]),
                "speedup": float(fields[4]),
            }
        )

if len(rows) < 3:
    raise SystemExit("Unable to parse benchmark rows from %s" % results)

csv_path = plots / "throughput.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=("kernel", "time_s", "melem_per_s", "mdof_per_s", "speedup"))
    writer.writeheader()
    writer.writerows(rows)

labels = [row["kernel"] for row in rows]
values = [row["melem_per_s"] for row in rows]
max_value = max(values) if values else 1.0
width = 760
height = 420
left = 100
bottom = 340
bar_width = 150
gap = 60
scale = 240.0 / max_value
svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
    '<rect width="100%" height="100%" fill="white"/>',
    '<text x="30" y="40" font-family="sans-serif" font-size="22" font-weight="600">TET4 Linear Elasticity Apply Throughput</text>',
    '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333" stroke-width="1"/>' % (left - 20, bottom, width - 60, bottom),
    '<text x="30" y="70" font-family="sans-serif" font-size="13" fill="#555">Higher is better. Source: benchmark_output.txt</text>',
]
colors = ["#2563eb", "#0f766e", "#9333ea", "#64748b"]
for i, (label, value) in enumerate(zip(labels, values)):
    x = left + i * (bar_width + gap)
    h = value * scale
    y = bottom - h
    svg.extend(
        [
            '<rect x="%.1f" y="%.1f" width="%d" height="%.1f" fill="%s"/>' % (x, y, bar_width, h, colors[i % len(colors)]),
            '<text x="%.1f" y="%.1f" font-family="sans-serif" font-size="14" text-anchor="middle">%.3f MElem/s</text>' % (x + bar_width / 2, y - 24, value),
            '<text x="%.1f" y="%.1f" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#555">%.3f MDOF/s</text>' % (x + bar_width / 2, y - 8, rows[i]["mdof_per_s"]),
            '<text x="%.1f" y="%d" font-family="sans-serif" font-size="13" text-anchor="middle">%s</text>' % (x + bar_width / 2, bottom + 25, label),
        ]
    )
svg.append("</svg>")
(plots / "throughput.svg").write_text("\n".join(svg))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(labels, values, color=["#2563eb", "#0f766e", "#9333ea", "#64748b"][:len(labels)])
    ax.set_ylabel("MElem/s")
    ax.set_title("TET4 Linear Elasticity Apply Throughput")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, value, "%.3f MElem/s\n%.3f MDOF/s" % (value, rows[i]["mdof_per_s"]), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "throughput.png", dpi=160)
    plt.close(fig)
except Exception as exc:
    (plots / "throughput_png_error.txt").write_text("%s: %s\n" % (type(exc).__name__, exc))

print("plots_dir %s" % plots)
print("csv %s" % csv_path)
print("svg %s" % (plots / "throughput.svg"))
png_path = plots / "throughput.png"
if png_path.exists():
    print("png %s" % png_path)
PY
