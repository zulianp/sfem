#!/usr/bin/env bash

set -e

source $CODE_DIR/merge_git_repos/sfem/venv/bin/activate
# export SFEM_PATH=$INSTALL_DIR/sfem

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH
export PATH=$SFEM_PATH/scripts/sfem/mesh/:$PATH
export PATH=$SFEM_PATH/scripts/sfem/grid/:$PATH
export PATH=$SFEM_PATH/scripts/sfem/sdf/:$PATH
export PATH=$SFEM_PATH/worflows/mech/:$PATH
export PATH=$CODE_DIR/merge_git_repos/sfem/data/benchmarks/meshes:$PATH

HERE=$PWD

rm -rf hex8_geometry
if [[ ! -d hex8_geometry ]]
then
	mkdir -p hex8_geometry
	cd hex8_geometry

	box_mesh.py box --cell_type=hex8 -x 10 -y 10 -z 10
	surf_type=quad4
	
	skin box skin_box
	raw_to_db.py skin_box skin_box.vtk

	set -x

	SFEM_DEBUG=1 create_sideset box  -0.001 0.5 0.5  0.8 	inlet
	SFEM_DEBUG=1 create_sideset box   1.001 0.5 0.5  0.8 	outlet

	raw_to_db.py inlet/surf 			inlet/surf.vtk 				--coords=box --cell_type=$surf_type
	raw_to_db.py outlet/surf 			outlet/surf.vtk 			--coords=box --cell_type=$surf_type
	raw_to_db.py box 					box.vtk 

	cd $HERE
fi

export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
$LAUNCH hyperelasticy hex8_geometry/box dirichlet_hex8.yaml hex8_output
raw_to_db.py hex8_geometry/box hex8_output.vtk -p 'hex8_output/out/*.raw' $EXTRA_OPTIONS
