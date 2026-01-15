#!/usr/bin/env bash

set -e

# source $CODE_DIR/merge_git_repos/sfem/venv/bin/activate
# export SFEM_PATH=$INSTALL_DIR/sfem

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PATH=$SFEM_PATH/bin:$PATH
export PATH=$SFEM_PATH/scripts/sfem/mesh/:$PATH
export PATH=$SFEM_PATH/scripts/sfem/grid/:$PATH
export PATH=$SFEM_PATH/scripts/sfem/sdf/:$PATH
export PATH=$SFEM_PATH/worflows/mech/:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

HERE=$PWD

rm -rf torus_geometry
if [[ ! -d torus_geometry ]]
then
	mkdir -p torus_geometry
	cd torus_geometry

	db_to_raw.py ../torus_cut_hex.vtk torus
	surf_type=quad4
	
	skin torus skin_torus
	raw_to_db.py skin_torus skin_torus.vtk

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

rm -rf torus_output
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
$LAUNCH hyperelasticy torus_geometry/torus dirichlet_torus.yaml torus_output
raw_to_db.py torus_output/mesh torus_output.vtk -p 'torus_output/out/*.raw' $EXTRA_OPTIONS

