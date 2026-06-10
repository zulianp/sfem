#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH

HERE=$PWD


rm -rf geometry_torus
if [[ ! -d geometry_torus ]]
then
	mkdir -p geometry_torus
	cd geometry_torus

	torus 5 2.5 torus.vtk --refinements=2 --order=2
	db_to_raw torus.vtk torus --select_elem_type=tetra10
	
	skin torus skin_torus
	raw_to_db skin_torus skin_torus.vtk

	set -x

	create_sideset torus   -7.5 0 0 0.997 	inlet
	create_sideset torus    2.5 0 0 0.99	outlet

	surface_from_sideset torus inlet  inlet/surf
	surface_from_sideset torus outlet outlet/surf

	raw_to_db inlet/surf  inlet/surf.vtk   --coords=torus
	raw_to_db outlet/surf outlet/surf.vtk  --coords=torus
	raw_to_db torus 	  torus.vtk 

	cd $HERE
fi

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

export SFEM_FIRST_LAME_PARAMETER=2
export SFEM_SHEAR_MODULUS=2

rm -rf output_torus
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
$LAUNCH hyperelasticy geometry_torus/torus dirichlet_torus.yaml output_torus

raw_to_db output_torus/mesh output_torus/output_torus.vtk -p 'output_torus/out/*.*' $EXTRA_OPTIONS


