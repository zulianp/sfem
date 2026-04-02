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

	sfc mesh mesh
	
	skin mesh mesh_surface
	raw_to_db mesh_surface mesh_surface.vtk
	cd $HERE
fi


echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"


export SFEM_EXECUTION_SPACE=device
# export SFEM_ELEMENT_REFINE_LEVEL=2
export SFEM_USE_SSGMG=0
export SFEM_DT=1
export SFEM_T_END=2
export SFEM_VERBOSE=1

# Use SIDESET-based Neumann input (expects parent.* in the given directory)
export SFEM_NEUMANN_SIDESET=geometry/mesh/sidesets/top
export SFEM_NEUMANN_COMPONENT=1
export SFEM_NEUMANN_VALUE=-5

rm -rf output
$LAUNCH kelvin_voigt_newmark geometry/mesh dirichlet.yaml output neumann.yaml
raw_to_db geometry/mesh output.xdmf -p "output/out/disp.0.*.*,output/out/disp.1.*.*,output/out/disp.2.*.*" --transient  --time_whole_txt=output/out/time.txt

