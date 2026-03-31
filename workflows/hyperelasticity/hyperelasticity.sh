#!/usr/bin/env bash

set -e

source $CODE_DIR/merge_git_repos/sfem/venv/bin/activate

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

rm -rf geometry
if [[ ! -d geometry ]]
then
	mkdir -p geometry
	cd geometry

	cylinder 0.1 1 mesh.vtk --order=2
	db_to_raw mesh.vtk mesh --select_elem_type=tetra10
	surf_type=tri6

	sfc mesh mesh
	
	skin mesh mesh_surface
	raw_to_db mesh_surface mesh_surface.vtk

	set -x

	create_sideset mesh -0.51 0 0  0.5 	inlet
	create_sideset mesh  0.51 0 0  0.5 	outlet

	surface_from_sideset mesh inlet  inlet/surf
	surface_from_sideset mesh outlet outlet/surf

	raw_to_db inlet/surf   inlet/surf.vtk 	--coords=mesh
	raw_to_db outlet/surf  outlet/surf.vtk  --coords=mesh
	raw_to_db mesh 		   mesh.vtk 

	cd $HERE
fi

export SFEM_ROTATE_SIDESET=geometry/outlet
export SFEM_ROTATE_ANGLE=12
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
