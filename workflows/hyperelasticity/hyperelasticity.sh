#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH

HERE=$PWD

rm -rf geometry
if [[ ! -d geometry ]]
then
	mkdir -p geometry
	cd geometry

	cylinder 0.1 1 mesh.vtk --order=2 --refinements=1
	db_to_raw mesh.vtk mesh --select_elem_type=tetra10

	sfc mesh mesh

	create_sideset mesh -0.51 0 0  0.5 	inlet
	create_sideset mesh  0.51 0 0  0.5 	outlet

	# Just for viz
	skin mesh mesh_surface
	surface_from_sideset mesh inlet  inlet/surf
	surface_from_sideset mesh outlet outlet/surf
	
	raw_to_db mesh_surface mesh_surface.vtk
	raw_to_db inlet/surf   inlet/surf.vtk 	--coords=mesh
	raw_to_db outlet/surf  outlet/surf.vtk  --coords=mesh
	raw_to_db mesh 		   mesh.vtk 

	cd $HERE
fi

export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
export SFEM_USE_PARTIAL_ASSEMBLY=1
export SFEM_ELEMENTS_PER_PACK=2048 
export SMESH_TRACE_FILE=output/hyperelasticity.trace.csv

rm -rf output

$LAUNCH playground hyperelasticity.yaml

cd output
raw_to_db mesh output.xdmf -p "out/disp.0.*.*,out/disp.1.*.*,out/disp.2.*.*" --transient --n_time_steps=25
cd $HERE