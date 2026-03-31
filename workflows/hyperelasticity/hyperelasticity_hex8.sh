#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH

HERE=$PWD

rm -rf geometry_hex8
if [[ ! -d geometry_hex8 ]]
then
	mkdir -p geometry_hex8
	cd geometry_hex8

	MESH_FACTOR=1
	cube HEX8 $((120 * MESH_FACTOR)) $((30 * MESH_FACTOR)) $((30 * MESH_FACTOR)) 0 0 0 4 1 1 box
	surf_type=quad4
	
	skin box skin_box
	raw_to_db skin_box skin_box.vtk

	set -x

	create_sideset box  -0.001 0.5 0.5  0.8 	inlet
	create_sideset box   4.001 0.5 0.5  0.8 	outlet

	surface_from_sideset box inlet  inlet/surf
	surface_from_sideset box outlet outlet/surf

	raw_to_db inlet/surf 			inlet/surf.vtk 				--coords=box --cell_type=$surf_type
	raw_to_db outlet/surf 			outlet/surf.vtk 			--coords=box --cell_type=$surf_type
	raw_to_db box 					box.vtk 

	cd $HERE
fi

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

export SFEM_ROTATE_SIDESET=geometry_hex8/outlet
export SFEM_ROTATE_ANGLE=4
export SFEM_ROTATE_STEPS=12
export SFEM_ROTATE_RCENTER_X=0
export SFEM_ROTATE_RCENTER_Y=0.5
export SFEM_ROTATE_RCENTER_Z=0.5

rm -rf output_hex8
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
export SFEM_ELEMENTS_PER_PACK=4096 
export SFEM_USE_PACKED_MESH=1
export SFEM_USE_PRECONDITIONER=0
export SFEM_ENABLE_LINE_SEARCH=0
export SFEM_OPERATOR=NeoHookeanOgdenPacked


$LAUNCH hyperelasticy geometry_hex8/box dirichlet_hex8.yaml output_hex8

raw_to_db output_hex8/mesh output_hex8.xdmf -p "output_hex8/out/disp.0.*.*,output_hex8/out/disp.1.*.*,output_hex8/out/disp.2.*.*" --transient --n_time_steps=$SFEM_ROTATE_STEPS 
