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

rm -rf blade_geometry
if [[ ! -d blade_geometry ]]
then
	mkdir -p blade_geometry
	cd blade_geometry

	db_to_raw.py ../Blade_hex.vtk blade
	surf_type=quad4
	
	skin blade skin_blade
	raw_to_db.py skin_blade skin_blade.vtk

	set -x

	SFEM_DEBUG=1 create_sideset blade   0.19 -0.5 -0.036   0.99 	inlet
	SFEM_DEBUG=1 create_sideset blade  -0.20  -0.5  0.0115  0.99 	outlet

	raw_to_db.py inlet/surf 			inlet/surf.vtk 				--coords=blade --cell_type=$surf_type
	raw_to_db.py outlet/surf 			outlet/surf.vtk 			--coords=blade --cell_type=$surf_type
	raw_to_db.py blade 					blade.vtk 

	cd $HERE
fi

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

rm -rf blade_output
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
$LAUNCH hyperelasticy blade_geometry/blade dirichlet_blade.yaml blade_output
raw_to_db.py blade_output/mesh blade_output.vtk -p 'blade_output/out/*.raw' $EXTRA_OPTIONS
