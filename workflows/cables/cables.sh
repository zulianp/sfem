#!/usr/bin/env bash

set -e

# source $CODE_DIR/merge_git_repos/sfem/venv/bin/activate
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

rm -rf cables_geometry
if [[ ! -d cables_geometry ]]
then
	mkdir -p cables_geometry
	cd cables_geometry

	db_to_raw.py /Users/patrickzulian/Desktop/code/im2ex/build/trab_hex.vtk cables #--select_elem_type=tetra
	# surf_type=tri3
	surf_type=quad4
	
	skin cables skin_cables
	raw_to_db.py skin_cables skin_cables.vtk

	set -x
	# create_sideset format: create_sideset <mesh> <x> <y> <z> <radius> <output_name>
	# Coordinates extracted from sspics screenshots (Use tesseract for OCR)
	SFEM_DEBUG=1 create_sideset cables 128 128 0  0.997 ex1
	SFEM_DEBUG=1 create_sideset cables 120 100 256 0.997 ex2

	# Note: Last screenshot (11.03.31.png) shows incomplete coordinates: 0.323563, 1.59174) - missing x coordinate

	raw_to_db.py ex1/surf 			ex1/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex2/surf 			ex2/surf.vtk 			--coords=cables --cell_type=$surf_type
	
	
	# raw_to_db.py cables 			cables.vtk 

	cd $HERE
fi

export SFEM_USE_PARTIAL_ASSEMBLY=1
export SFEM_USE_COMPRESSION=1
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
$LAUNCH hyperelasticy cables_geometry/cables dirichlet.yaml output
raw_to_db.py cables_geometry/cables output.vtk -p 'output/out/*.raw' $EXTRA_OPTIONS
