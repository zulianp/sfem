#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../../build/sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH
export PATH=$SCRIPTPATH/../../../matrix.io:$PATH

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

mesh=mesh
simulation_mesh=simulation_mesh
refined_mesh=refined_mesh


if [[ -d "$simulation_mesh" ]]
then
	echo "Reusing $simulation_mesh"
else
	tet_mesh.py
	# create_square.sh 6
	# create_sphere.sh 4
	# refine $mesh temp1
	# refine temp1 temp2

	mesh_p1_to_p2 $mesh $simulation_mesh
	refine $mesh $refined_mesh
	
fi

mkdir -p linear_system linear_system_macro

export SFEM_HANDLE_NEUMANN=0 
export SFEM_HANDLE_DIRICHLET=0 	

assemble3 $refined_mesh linear_system
SFEM_USE_MACRO=1 $LAUNCH assemble3 $simulation_mesh linear_system_macro

MATRIXIO_DENSE_OUTPUT=1 print_crs linear_system/rowptr.raw linear_system/colidx.raw linear_system/values.raw int int double
MATRIXIO_DENSE_OUTPUT=1 print_crs linear_system_macro/rowptr.raw linear_system_macro/colidx.raw linear_system_macro/values.raw int int double

# eval_nodal_function.py "100 * x*x + y" $simulation_mesh/x.raw $simulation_mesh/y.raw  $simulation_mesh/z.raw linear_system/rhs.raw
# eval_nodal_function.py "x+y+z" $simulation_mesh/x.raw $simulation_mesh/y.raw  $simulation_mesh/z.raw linear_system/rhs.raw

# spmv 1 0 linear_system 		 linear_system/rhs.raw test.raw
# spmv 1 0 linear_system_macro linear_system/rhs.raw test_macro.raw
