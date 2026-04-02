#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH

HERE=$PWD

rm -rf geometry_blade
if [[ ! -d geometry_blade ]]
then
	mkdir -p geometry_blade
	cd geometry_blade

	db_to_raw ../Blade_hex.vtk blade
	surf_type=quad4
	
	create_sideset blade   0.19 -0.5 -0.036   0.99 	inlet
	create_sideset blade  -0.20  -0.5  0.0115  0.99 outlet

	# Just for viz
	skin blade skin_blade
	surface_from_sideset blade inlet  inlet/surf
	surface_from_sideset blade outlet outlet/surf

	raw_to_db skin_blade 	skin_blade.vtk 	--cell_type=$surf_type
	raw_to_db inlet/surf 	inlet/surf.vtk  --cell_type=$surf_type  --coords=blade 
	raw_to_db outlet/surf 	outlet/surf.vtk --cell_type=$surf_type  --coords=blade 
	raw_to_db blade 		blade.vtk 		--cell_type=$surf_type

	cd $HERE
fi

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"
export SMESH_TRACE_FILE=output_blade/hyperelasticity.trace.csv

rm -rf output_blade
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
$LAUNCH hyperelasticy geometry_blade/blade dirichlet_blade.yaml output_blade
raw_to_db output_blade/mesh output_blade.vtk -p 'output_blade/out/*.*' $EXTRA_OPTIONS
