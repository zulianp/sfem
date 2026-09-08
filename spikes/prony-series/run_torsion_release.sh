#!/usr/bin/env bash
# End-to-end scenario: build a HEX8 beam, twist it, hold it while the Prony series relaxes,
# release it, and check that the result behaves like a viscoelastic solid.
#
# Everything is produced under $WORKDIR, including a copy of the case file, so that the
# relative paths in the case resolve identically for this driver and for SFEM's own YAML
# parser -- SFEM resolves the inline dirichlet paths against the working directory, so the
# driver is invoked from $WORKDIR.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd -- "$HERE/../.." >/dev/null 2>&1 && pwd -P)"
PYTHON="${PRONY_PYTHON:-$ROOT_DIR/venv/bin/python}"
EXE="${PRONY_EXE:-$HERE/build/prony_visco_torsion}"
WORKDIR="${1:-$HERE/output}"

if [[ ! -x "$EXE" ]]; then
    echo "missing executable: $EXE" >&2
    echo "configure and build the spike first, e.g." >&2
    echo "  cmake -S $HERE -B $HERE/build -DSFEM_DIR=<prefix>/lib/cmake" >&2
    echo "  cmake --build $HERE/build -j" >&2
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "missing python: $PYTHON (set PRONY_PYTHON)" >&2
    exit 1
fi

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

# The viscoelastic Mooney-Rivlin kernels are HEX8 only, so the beam is hexahedral. The mesh
# lands in $WORKDIR/mesh with left/right sidesets on the two x faces, which is what the case
# file refers to.
"$PYTHON" "$ROOT_DIR/python/sfem/mesh/box_mesh.py" "$WORKDIR/mesh" \
    --cell_type=HEX8 \
    -x "${PRONY_NX:-16}" \
    -y "${PRONY_NY:-5}" \
    -z "${PRONY_NZ:-5}" \
    --width  "${PRONY_LENGTH:-1.0}" \
    --height "${PRONY_HEIGHT:-0.2}" \
    --depth  "${PRONY_DEPTH:-0.2}"

cp "$HERE/cases/torsion_release.yaml" "$WORKDIR/"
cp "$HERE/cases/torsion_hold.yaml"    "$WORKDIR/"

cd "$WORKDIR"

# Single-threaded by default, and that is not a concession -- it is the faster setting here by
# an order of magnitude. Every CG iteration calls the constraint handling, whose loops run over
# a few hundred boundary nodes; at this problem size an OpenMP team costs far more to start and
# join than the loop costs to run. Measured on a 2160-dof beam (720 nodes, 475 HEX8
# elements) over 12 steps and 72,943 CG iterations: 3.07 s on one thread, 9.66 s on four, 27.46 s on eight, with system
# time (thread barriers) accounting for the whole difference. Raise it for a mesh large enough
# to pay for the teams.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

PRONY_EXE="$EXE" PRONYRUN_DIR="$WORKDIR/logs" \
    "$HERE/tools/pronyrun.sh" torsion_release torsion_release.yaml

# The control run is what separates relaxation from release: same material, same twist, never
# let go. Off by default because it doubles the cost of the scenario.
if [[ "${PRONY_RUN_CONTROL:-0}" == "1" ]]; then
    PRONY_EXE="$EXE" PRONYRUN_DIR="$WORKDIR/logs" \
        "$HERE/tools/pronyrun.sh" torsion_hold torsion_hold.yaml
fi

# ParaView-ready transient files. The driver already writes out/disp.N.*.* and out/time.txt in
# exactly the layout this script expects, so it is reused unchanged rather than reimplemented.
XDMF_WRITER="$ROOT_DIR/workflows/hyperelasticity_bdf2/write_transient_xdmf.sh"
if [[ -x "$XDMF_WRITER" ]]; then
    "$XDMF_WRITER" "$WORKDIR/results_release/mesh" "$WORKDIR/results_release" \
        "$WORKDIR/results_release/output.xdmf" || echo "[warn] XDMF export failed; fields are still in results_release/out" >&2
fi

VALIDATE_ARGS=("$WORKDIR/results_release/history.csv" --case "$WORKDIR/torsion_release.yaml")
if [[ "${PRONY_RUN_CONTROL:-0}" == "1" ]]; then
    VALIDATE_ARGS+=(--control "$WORKDIR/results_hold/history.csv")
fi

"$PYTHON" "$HERE/tools/validate_torsion_release.py" "${VALIDATE_ARGS[@]}"

echo
echo "Prony-series torsion-with-release scenario completed:"
echo "  history:  $WORKDIR/results_release/history.csv"
echo "  fields:   $WORKDIR/results_release/out"
echo "  paraview: $WORKDIR/results_release/output.xdmf"
echo "  raw logs: $WORKDIR/logs"
