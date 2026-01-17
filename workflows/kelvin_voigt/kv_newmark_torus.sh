#!/usr/bin/env bash

set -e


# export SFEM_PATH=/users/hyang/ws/sfem_github/sfem/build_release
if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Activate venv to ensure meshio is available
# source /users/hyang/ws/sfem_github/sfem/venv/bin/activate

# Add venv's site-packages to PYTHONPATH as fallback
# export PYTHONPATH=/users/hyang/ws/sfem_github/sfem/venv/lib/python3.12/site-packages:$PYTHONPATH

export PATH=$SFEM_PATH:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/sdf:$PATH
export PATH=$SCRIPTPATH/../../workflows/mech:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH



export SFEM_EXECUTION_SPACE=device
# export SFEM_ELEMENT_REFINE_LEVEL=2
export SFEM_DT=1
export SFEM_T_END=2
export SFEM_VERBOSE=1




HERE=$PWD

# rm -rf torus_geometry
if [[ ! -d torus_geometry ]]
then
	mkdir -p torus_geometry
	cd torus_geometry

	python3 $SCRIPTPATH/../../python/sfem/mesh/db_to_raw.py ../../hyperelasticity/torus_cut_hex.vtk torus
	surf_type=quad4
	
	skin torus skin_torus
	python3 $SCRIPTPATH/../../python/sfem/mesh/raw_to_db.py skin_torus skin_torus.vtk

	set -x

	SFEM_DEBUG=1 create_sideset torus   4.96663 0.0306772 1.18258  0.998 	inlet
	SFEM_DEBUG=1 create_sideset torus   4.98568 9.98184   1.10985  0.998 	outlet

	raw_to_db.py inlet/surf 			inlet/surf.vtk 				--coords=torus --cell_type=$surf_type
	raw_to_db.py outlet/surf 			outlet/surf.vtk 			--coords=torus --cell_type=$surf_type
	raw_to_db.py torus 					torus.vtk 

	cd $HERE
fi

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

rm -rf torus_kv_output


# Use SIDESET-based Neumann input (expects parent.raw in the given directory)
export SFEM_NEUMANN_SIDESET=torus_geometry/outlet
export SFEM_NEUMANN_COMPONENT=1
export SFEM_NEUMANN_VALUE=-5


$LAUNCH kelvin_voigt_newmark torus_geometry/torus dirichlet_torus_kv.yaml torus_kv_output neumann_torus_kv.yaml
raw_to_db.py torus_kv_output/mesh torus_kv_output.xdmf -p "torus_kv_output/out/disp.0.*.raw,torus_kv_output/out/disp.1.*.raw,torus_kv_output/out/disp.2.*.raw" --transient  --time_whole_txt=torus_kv_output/out/time.txt
