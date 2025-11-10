#!/usr/bin/env bash

set -e

# source $CODE_DIR/merge_git_repos/sfem/venv/bin/activate
# export SFEM_PATH=$INSTALL_DIR/sfem

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PATH=$SFEM_PATH/bin:$PATH
export PATH=$SFEM_PATH/scripts/sfem/mesh/:$PATH
export PATH=$SFEM_PATH/scripts/sfem/grid/:$PATH
export PATH=$SFEM_PATH/scripts/sfem/sdf/:$PATH
export PATH=$SFEM_PATH/worflows/mech/:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

HERE=$PWD

# rm -rf hex8_geometry
if [[ ! -d hex8_geometry ]]
then
	mkdir -p hex8_geometry
	cd hex8_geometry

	MESH_FACTOR=1
	box_mesh.py box --cell_type=hex8 -x $((120 * MESH_FACTOR)) -y $((30 * MESH_FACTOR)) -z $((30 * MESH_FACTOR)) --width=4
	surf_type=quad4
	
	skin box skin_box
	raw_to_db.py skin_box skin_box.vtk

	set -x

	SFEM_DEBUG=1 create_sideset box  -0.001 0.5 0.5  0.8 	inlet
	SFEM_DEBUG=1 create_sideset box   4.001 0.5 0.5  0.8 	outlet

	raw_to_db.py inlet/surf 			inlet/surf.vtk 				--coords=box --cell_type=$surf_type
	raw_to_db.py outlet/surf 			outlet/surf.vtk 			--coords=box --cell_type=$surf_type
	raw_to_db.py box 					box.vtk 

	cd $HERE
fi

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

export SFEM_ROTATE_SIDESET=hex8_geometry/outlet
# export SFEM_ROTATE_ANGLE=3.14
# export SFEM_ROTATE_STEPS=20
export SFEM_ROTATE_ANGLE=0
export SFEM_ROTATE_STEPS=30
export SFEM_ROTATE_RCENTER_X=0
export SFEM_ROTATE_RCENTER_Y=0.5
export SFEM_ROTATE_RCENTER_Z=0.5

rm -rf hex8_output
export SFEM_NEOHOOKEAN_OGDEN_USE_AOS=1
export SFEM_ELEMENTS_PER_PACK=4096 
export SFEM_USE_PACKED_MESH=1
export SFEM_USE_PRECONDITIONER=0
export SFEM_ENABLE_LINE_SEARCH=0
export SFEM_OPERATOR=NeoHookeanOgdenActiveStrainPacked
export SFEM_ACTIVE_STRAIN_RADIUS=1 
export SFEM_LSOLVE_RTOL=1e-4

# xctrace record --template 'Time Profiler'  --launch -- $SFEM_PATH/bin/hyperelasticy hex8_geometry/box dirichlet_hex8.yaml hex8_output
$LAUNCH hyperelasticy hex8_geometry/box dirichlet_active_strain_hex8.yaml hex8_output
# raw_to_db.py hex8_output/mesh hex8_output.vtk -p 'hex8_output/out/*.raw' $EXTRA_OPTIONS

raw_to_db.py hex8_output/mesh hex8_output.xdmf -p "hex8_output/out/disp.0.*.raw,hex8_output/out/disp.1.*.raw,hex8_output/out/disp.2.*.raw" --transient --n_time_steps=$((SFEM_ROTATE_STEPS  + 1))
