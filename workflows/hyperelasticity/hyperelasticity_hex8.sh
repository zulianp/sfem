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

	create_sideset box  -0.001 0.5 0.5  0.8 	inlet
	create_sideset box   4.001 0.5 0.5  0.8 	outlet

	# Just for viz
	skin box skin_box
	surface_from_sideset box inlet  inlet/surf
	surface_from_sideset box outlet outlet/surf

	raw_to_db skin_box skin_box.vtk
	raw_to_db inlet/surf 			inlet/surf.vtk 				--coords=box --cell_type=$surf_type
	raw_to_db outlet/surf 			outlet/surf.vtk 			--coords=box --cell_type=$surf_type
	raw_to_db box 					box.vtk 

	cd $HERE
fi

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"


rm -rf output_hex8
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
export SFEM_ELEMENTS_PER_PACK=4096 
export SMESH_TRACE_FILE=output_hex8/hyperelasticity.trace.csv

$LAUNCH playground hyperelasticity_hex8.yaml

cd output_hex8
raw_to_db mesh output_hex8.xdmf -p "out/disp.0.*.*,out/disp.1.*.*,out/disp.2.*.*" --transient --n_time_steps=13 
cd $HERE