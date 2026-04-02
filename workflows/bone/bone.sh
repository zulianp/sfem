#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH

HERE=$PWD

rm -rf bone_geometry
if [[ ! -d bone_geometry ]]
then
	mkdir -p bone_geometry
	cd bone_geometry

	exodusII_to_raw ../bone.e bone
	surf_type=tri3
	
	create_sideset bone 386 614 0      0.99 	inlet
	create_sideset bone 340 424 373.92 0.99	outlet

	# Just for viz
	skin bone skin_bone
	surface_from_sideset bone inlet  inlet/surf
	surface_from_sideset bone outlet outlet/surf

	raw_to_db skin_bone skin_bone.vtk
	raw_to_db inlet/surf 			inlet/surf.vtk 				--coords=bone --cell_type=$surf_type
	raw_to_db outlet/surf 			outlet/surf.vtk 			--coords=bone --cell_type=$surf_type
	raw_to_db bone 					bone.vtk 

	cd $HERE
fi

export SFEM_USE_PARTIAL_ASSEMBLY=1
export SFEM_USE_COMPRESSION=1
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
$LAUNCH hyperelasticy bone_geometry/bone dirichlet.yaml output
raw_to_db bone_geometry/bone output.vtk -p 'output/out/*.*' $EXTRA_OPTIONS
