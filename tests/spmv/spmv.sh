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

LAUNCH="mpiexec -np 8"
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=true 

# export SFEM_REPEAT=20

# LAUNCH=""

source venv/bin/activate

mesh=mesh
mesh_sorted=ref2



if [[ -d "$mesh_sorted" ]]
then
	echo "Reusing mesh: $mesh_sorted"
else
	create_sphere.sh 4
	refine mesh ref1
	sfc ref1 sfc1
	refine sfc1 ref2
	mesh_p1_to_p2 sfc1 p2


	touch $mesh_sorted/zd.raw
	touch $mesh_sorted/on.raw

	mkdir -p linear_system
	assemble $mesh_sorted linear_system

	eval_nodal_function.py "x*x + y*y" $mesh_sorted/x.raw $mesh_sorted/y.raw  $mesh_sorted/z.raw linear_system/rhs.raw
	# eval_nodal_function.py "x*0 + 1" $mesh_sorted/x.raw $mesh_sorted/y.raw  $mesh_sorted/z.raw linear_system/rhs.raw
fi

spmv 	1 0 linear_system linear_system/rhs.raw test.raw
cuspmv 	1 0 linear_system linear_system/rhs.raw test.raw

lapl_matrix_free $mesh_sorted 1 linear_system/rhs.raw mf_test.raw
SFEM_USE_MACRO=1 lapl_matrix_free p2 1 linear_system/rhs.raw macro_test.raw
SFEM_USE_MACRO=0 lapl_matrix_free p2 1 linear_system/rhs.raw p2_test.raw

raw_to_db.py $mesh_sorted mf_out.vtk --point_data="linear_system/rhs.raw,mf_test.raw,macro_test.raw,test.raw,p2_test.raw"

deactivate