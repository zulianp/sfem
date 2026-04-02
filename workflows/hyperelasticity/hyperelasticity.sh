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

export SFEM_ROTATE_SIDESET=geometry/outlet
export SFEM_ROTATE_ANGLE=7.5
export SFEM_ROTATE_STEPS=24
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
export SFEM_USE_PARTIAL_ASSEMBLY=1
export SFEM_ELEMENTS_PER_PACK=2048 
export SFEM_USE_PACKED_MESH=1
export SFEM_USE_PRECONDITIONER=0
export SFEM_ENABLE_LINE_SEARCH=0
rm -rf output

$LAUNCH hyperelasticy geometry/mesh dirichlet.yaml output
raw_to_db output/mesh output.xdmf -p "output/out/disp.0.*.*,output/out/disp.1.*.*,output/out/disp.2.*.*" --transient --n_time_steps=$(( SFEM_ROTATE_STEPS + 1 ))
