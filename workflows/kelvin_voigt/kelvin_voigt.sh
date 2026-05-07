#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH:$SFEM_PATH/bin:$SFEM_PATH/external/smesh:$PATH

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

if ! command -v raw_to_db >/dev/null 2>&1; then
	if python3 -c 'import meshio' >/dev/null 2>&1; then
		raw_to_db() {
			python3 "$REPO_ROOT/python/sfem/mesh/raw_to_db.py" "$@"
		}
	else
		raw_to_db() {
			echo "Skipping raw_to_db $* (raw_to_db command not found and Python meshio is not installed)"
		}
	fi
fi

HERE=$PWD

rm -rf geometry
if [[ ! -d geometry ]]
then
	mkdir -p geometry
	cd geometry

	cube HEX8 10 10 10 0 0 0 1 1 1 mesh
	surf_type=quad4

	raw_to_db mesh mesh.vtk
	
	set -x

	surface_from_sideset mesh mesh/sidesets/top    mesh/sidesets/top/surf
	surface_from_sideset mesh mesh/sidesets/bottom mesh/sidesets/bottom/surf

	raw_to_db mesh/sidesets/top/surf    top.vtk    --coords=mesh --cell_type=$surf_type
	raw_to_db mesh/sidesets/bottom/surf bottom.vtk --coords=mesh --cell_type=$surf_type
	set +x
	cd $HERE
fi

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

export SFEM_EXECUTION_SPACE=${SFEM_EXECUTION_SPACE:-host}
export SFEM_ELEMENT_REFINE_LEVEL=${SFEM_ELEMENT_REFINE_LEVEL:-0}
export SFEM_USE_SSGMG=${SFEM_USE_SSGMG:-0}
export SFEM_DT=${SFEM_DT:-0.01}
export SFEM_T_END=${SFEM_T_END:-2}
export SFEM_VERBOSE=${SFEM_VERBOSE:-0}
export SFEM_BODY_FORCE_X=${SFEM_BODY_FORCE_X:-0.1}
export SFEM_BODY_FORCE_Y=${SFEM_BODY_FORCE_Y:-0}
export SFEM_BODY_FORCE_Z=${SFEM_BODY_FORCE_Z:-0}
export SFEM_EXPORT_FREQ=${SFEM_EXPORT_FREQ:-1}
export LAUNCH=${LAUNCH:-}

NEUMANN_CONDITIONS=${SFEM_NEUMANN_CONDITIONS:-NONE}

export SMESH_TRACE_FILE=output/kv.trace.csv

rm -rf output
$LAUNCH kelvin_voigt_newmark geometry/mesh dirichlet.yaml "$NEUMANN_CONDITIONS" output

cd output
raw_to_db ../geometry/mesh output.xdmf -p "out/disp.0.*.*,out/disp.1.*.*,out/disp.2.*.*" --transient  --time_whole_txt=out/time.txt
cd $HERE
