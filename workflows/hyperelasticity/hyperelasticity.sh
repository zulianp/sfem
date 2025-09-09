#!/usr/bin/env bash

set -e

source $CODE_DIR/merge_git_repos/sfem/venv/bin/activate
export SFEM_PATH=$INSTALL_DIR/sfem

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

# REF=7
REF=8
IR=0.4
OR=0.6
H=1
L=0
R=$H
MID=-0.55

nx=$(( REF * 8 ))
nr=$(( REF * 1 ))
nz=$(( REF * 8 ))

rm -rf aorta_geometry
if [[ ! -d aorta_geometry ]]
then
	mkdir -p aorta_geometry
	cd aorta_geometry

	# create_ring_mesh $IR $OR $nr $nx ring
	# mv ring/x0.raw ring/x.raw
	# mv ring/x1.raw ring/y.raw
	# mv ring/x2.raw ring/z.raw
	# db_to_raw.py $HERE/tet4.vtk aorta
	# surf_type=tri3
	cylinder.py aorta.vtk 2
	db_to_raw.py aorta.vtk aorta --select_elem_type=tetra
	surf_type=tri3
	
	skin aorta skin_aorta
	raw_to_db.py skin_aorta skin_aorta.vtk

	# SFEM_TRANSLATE_Z=$L hex8_extrude_mesh ring $H $nz aorta
	# mv aorta/x0.raw aorta/x.raw
	# mv aorta/x1.raw aorta/y.raw
	# mv aorta/x2.raw aorta/z.raw
	# surf_type=quad4
	

	# SFEM_TRANSLATE_Z=$L hex8_extrude_mesh ring $H $nz hex8_aorta
	# hex8_to_tet4 hex8_aorta aorta
	# mv hex8_aorta/x0.raw aorta/x.raw
	# mv hex8_aorta/x1.raw aorta/y.raw
	# mv hex8_aorta/x2.raw aorta/z.raw
	# surf_type=tri3

	rm -rf ring

	set -x

	# SFEM_DEBUG=1 create_sideset aorta 0    0     0.1 0.2 	contact_boundary
	# SFEM_DEBUG=1 create_sideset aorta $OR  $OR   0.1 0.2 	outer_boundary
	SFEM_DEBUG=1 $LAUNCH create_sideset aorta -0.51 0 0  0.8 	inlet
	SFEM_DEBUG=1 create_sideset aorta  0.51 0 0  0.8 	outlet

	# raw_to_db.py contact_boundary/surf 	contact_boundary/surf.vtk 	--coords=aorta --cell_type=tri3
	# raw_to_db.py outer_boundary/surf 	outer_boundary/surf.vtk 	--coords=aorta --cell_type=tri3
	raw_to_db.py inlet/surf 			inlet/surf.vtk 				--coords=aorta --cell_type=$surf_type
	raw_to_db.py outlet/surf 			outlet/surf.vtk 			--coords=aorta --cell_type=$surf_type
	raw_to_db.py aorta 					aorta.vtk 

	cd $HERE
fi


$LAUNCH hyperelasticy aorta_geometry/aorta dirichlet.yaml output
raw_to_db.py aorta_geometry/aorta output.vtk -p 'output/out/*.raw' $EXTRA_OPTIONS
