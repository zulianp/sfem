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

	cube HEX8 10 10 10 0 0 0 1 1 1 mesh
	surf_type=quad4

	raw_to_db mesh mesh.vtk
	
	set -x

	surface_from_sideset mesh mesh/sidesets/top    mesh/sidesets/top/surf
	surface_from_sideset mesh mesh/sidesets/bottom mesh/sidesets/bottom/surf

	raw_to_db mesh/sidesets/top/surf    top.vtk    --coords=mesh --cell_type=$surf_type
	raw_to_db mesh/sidesets/bottom/surf bottom.vtk --coords=mesh --cell_type=$surf_type
	cd $HERE
fi

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

export SFEM_EXECUTION_SPACE=device
# export SFEM_ELEMENT_REFINE_LEVEL=2
export SFEM_USE_SSGMG=0
export SFEM_DT=0.01
export SFEM_T_END=2
export SFEM_VERBOSE=0

export SMESH_TRACE_FILE=output/kv.trace.csv

rm -rf output
$LAUNCH kelvin_voigt_newmark geometry/mesh dirichlet.yaml neumann.yaml output 

cd output
raw_to_db ../geometry/mesh output.xdmf -p "out/disp.0.*.*,out/disp.1.*.*,out/disp.2.*.*" --transient  --time_whole_txt=out/time.txt
cd $HERE
