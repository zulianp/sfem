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

rm -rf aorta_geometry
if [[ ! -d aorta_geometry ]]
then
	mkdir -p aorta_geometry
	cd aorta_geometry


	cylinder.py aorta.vtk 2
	db_to_raw.py aorta.vtk aorta --select_elem_type=tetra
	sfc aorta aorta
	mesh_p1_to_p2 aorta aorta
	surf_type=tri6
	
	skin aorta skin_aorta
	# raw_to_db.py skin_aorta skin_aorta.vtk

	set -x

	SFEM_DEBUG=1 create_sideset aorta -0.51 0 0  0.8 	inlet
	SFEM_DEBUG=1 create_sideset aorta  0.51 0 0  0.8 	outlet

	# raw_to_db.py inlet/surf 			inlet/surf.vtk 				--coords=aorta --cell_type=$surf_type
	# raw_to_db.py outlet/surf 			outlet/surf.vtk 			--coords=aorta --cell_type=$surf_type
	# raw_to_db.py aorta 					aorta.vtk 

	cd $HERE
fi

export SFEM_ROTATE_SIDESET=aorta_geometry/outlet
# export SFEM_ROTATE_ANGLE=6
# export SFEM_ROTATE_STEPS=31
export SFEM_ROTATE_ANGLE=5
export SFEM_ROTATE_STEPS=60
# export SFEM_ROTATE_ANGLE=1.2
# export SFEM_ROTATE_STEPS=10
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
export SFEM_USE_PARTIAL_ASSEMBLY=1

rm -rf output

$LAUNCH hyperelasticy aorta_geometry/aorta dirichlet.yaml output
# raw_to_db.py aorta_geometry/aorta output.vtk -p 'output/out/*.raw' $EXTRA_OPTIONS

raw_to_db.py output/mesh output.xdmf -p "output/out/disp.0.*.raw,output/out/disp.1.*.raw,output/out/disp.2.*.raw" --transient --n_time_steps=$SFEM_ROTATE_STEPS 
