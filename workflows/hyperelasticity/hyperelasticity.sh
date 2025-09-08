#!/usr/bin/env bash

set -e

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

HERE=$PWD

REF=7
# REF=3
IR=0.51
OR=0.61
H=10
L=-2.5
R=7.5
MID=0.56

nx=$(( REF * 20 ))
nr=$(( REF * 1 ))
nz=$(( REF * 25 ))

# rm -rf aorta_geometry
if [[ ! -d aorta_geometry ]]
then
	mkdir -p aorta_geometry
	cd aorta_geometry

	create_ring_mesh $IR $OR $nr $nx ring
	mv ring/x0.raw ring/x.raw
	mv ring/x1.raw ring/y.raw
	mv ring/x2.raw ring/z.raw

	SFEM_TRANSLATE_Z=$L hex8_extrude_mesh ring $H $nz hex8_aorta

	hex8_to_tet4 hex8_aorta aorta
	mv hex8_aorta/x0.raw aorta/x.raw
	mv hex8_aorta/x1.raw aorta/y.raw
	mv hex8_aorta/x2.raw aorta/z.raw

	rm -rf ring

	set -x

	SFEM_DEBUG=1 create_sideset aorta 0    0     0.1 0.2 	contact_boundary
	SFEM_DEBUG=1 create_sideset aorta $OR  $OR   0.1 0.2 	outer_boundary
	SFEM_DEBUG=1 create_sideset aorta $MID $MID  $L  0.999 	inlet
	SFEM_DEBUG=1 create_sideset aorta $MID $MID  $R  0.999 	outlet

	raw_to_db.py contact_boundary/surf 	contact_boundary/surf.vtk 	--coords=aorta --cell_type=tri3
	raw_to_db.py outer_boundary/surf 	outer_boundary/surf.vtk 	--coords=aorta --cell_type=tri3
	raw_to_db.py inlet/surf 			inlet/surf.vtk 				--coords=aorta --cell_type=tri3
	raw_to_db.py outlet/surf 			outlet/surf.vtk 			--coords=aorta --cell_type=tri3
	raw_to_db.py aorta 					aorta.vtk 

	cd $HERE
fi


$LAUNCH hyperelasticy aorta_geometry/aorta dirichlet.yaml output
raw_to_db.py aorta_geometry/aorta output.vtk -p 'output/out/*.raw' $EXTRA_OPTIONS
