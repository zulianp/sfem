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

# rm -rf cables_geometry
# if [[ ! -d cables_geometry ]]
# then
# 	mkdir -p cables_geometry
	cd cables_geometry

# 	db_to_raw.py /Users/patrickzulian/Desktop/code/im2ex/test.vtk  cables #--select_elem_type=tetra
	surf_type=tri3
	
# 	skin cables skin_cables
# 	raw_to_db.py skin_cables skin_cables.vtk

	set -x
	# create_sideset format: create_sideset <mesh> <x> <y> <z> <radius> <output_name>
	# Coordinates extracted from sspics screenshots (Use tesseract for OCR)
	SFEM_DEBUG=1 create_sideset cables -0.419534 0.279386 -1.59475 0.997 ex1
	SFEM_DEBUG=1 create_sideset cables 0.364017 -0.369657 -1.59476 0.997 ex2
	SFEM_DEBUG=1 create_sideset cables -0.0575283 -0.358242 1.5893 0.997 ex3
	SFEM_DEBUG=1 create_sideset cables -0.451914 0.147346 1.5845 0.997 ex4
	SFEM_DEBUG=1 create_sideset cables -0.239819 0.298883 1.59079 0.997 ex5
	SFEM_DEBUG=1 create_sideset cables 0.0546479 0.310284 1.58601 0.997 ex6
	SFEM_DEBUG=1 create_sideset cables 0.329973 0.193941 1.58156 0.997 ex7
	SFEM_DEBUG=1 create_sideset cables 0.510738 0.01398 1.59447 0.997 ex8
	SFEM_DEBUG=1 create_sideset cables 0.458526 -0.182621 1.59446 0.997 ex9
	SFEM_DEBUG=1 create_sideset cables 0.171758 -0.32476 -1.59469 0.997 ex10
	SFEM_DEBUG=1 create_sideset cables 0.333714 -0.0856002 -1.59329 0.997 ex11
	SFEM_DEBUG=1 create_sideset cables 0.122607 0.11743 -1.58441 0.997 ex12
	SFEM_DEBUG=1 create_sideset cables -0.144946 0.300158 -1.58977 0.997 ex13
	SFEM_DEBUG=1 create_sideset cables -0.323664 0.0648472 -1.59343 0.997 ex14
	SFEM_DEBUG=1 create_sideset cables -0.0893154 -0.111871 -1.58758 0.997 ex15
	# Note: Last screenshot (11.03.31.png) shows incomplete coordinates: 0.323563, 1.59174) - missing x coordinate

	raw_to_db.py ex1/surf 			ex1/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex2/surf 			ex2/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex3/surf 			ex3/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex4/surf 			ex4/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex5/surf 			ex5/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex6/surf 			ex6/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex7/surf 			ex7/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex8/surf 			ex8/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex9/surf 			ex9/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex10/surf 			ex10/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex11/surf 			ex11/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex12/surf 			ex12/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex13/surf 			ex13/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex14/surf 			ex14/surf.vtk 			--coords=cables --cell_type=$surf_type
	raw_to_db.py ex15/surf 			ex15/surf.vtk 			--coords=cables --cell_type=$surf_type
	
	# raw_to_db.py cables 			cables.vtk 

	cd $HERE
# fi

# export SFEM_USE_PARTIAL_ASSEMBLY=1
# export SFEM_USE_COMPRESSION=1
# export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
# $LAUNCH hyperelasticy cables_geometry/cables dirichlet.yaml output
# raw_to_db.py cables_geometry/cables output.vtk -p 'output/out/*.raw' $EXTRA_OPTIONS
