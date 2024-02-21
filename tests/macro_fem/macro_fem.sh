#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../../workflows/sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../..:$PATH
export PATH=$SCRIPTPATH/../../python/sfem:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/utils:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

# export OMP_NUM_THREADS=16
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 

# export SFEM_REPEAT=40
export SFEM_REPEAT=1

# LAUNCH=""

source venv/bin/activate

mesh=mesh
simulation_mesh=simulation_mesh
refined_mesh=refined_mesh


if [[ -d "$simulation_mesh" ]]
then
	echo "Reusing $simulation_mesh"
else
	# create_square.sh 6
	create_sphere.sh 3
	refine $mesh temp1
	refine temp1 temp2
	mesh_p1_to_p2 temp2 $simulation_mesh
	refine temp2 $refined_mesh

	mkdir -p linear_system
	eval_nodal_function.py "x*x" $simulation_mesh/x.raw $simulation_mesh/y.raw  $simulation_mesh/z.raw linear_system/rhs.raw
	SFEM_HANDLE_NEUMANN=0 SFEM_HANDLE_DIRICHLET=0 assemble $refined_mesh linear_system
fi

spmv 1 0 linear_system linear_system/rhs.raw test.raw
lumped_mass_inv $refined_mesh test.raw test_out.raw

SFEM_USE_OPT=1 macro_element_apply $simulation_mesh linear_system/rhs.raw mea_test.raw
lumped_mass_inv $refined_mesh mea_test.raw mf_out.raw
raw_to_db.py $refined_mesh mf_out.vtk --point_data="linear_system/rhs.raw,mea_test.raw,mf_out.raw,test.raw,test_out.raw"

deactivate
