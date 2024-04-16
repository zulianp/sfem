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
# export OMP_NUM_THREADS=16
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=true 
export SFEM_REPEAT=10

mesh=mesh
ref2=ref2

if [[ -d "$ref2" ]]
then
	echo "Reusing mesh: $ref2"
else
	create_sphere.sh 2
	refine mesh ref1
	sfc ref1 sfc1
	refine sfc1 ref2
	mesh_p1_to_p2 sfc1 p2

	touch $ref2/zd.raw
	touch $ref2/on.raw

	mkdir -p linear_system
	assemble $ref2 linear_system

	eval_nodal_function.py "x*x + y*y" $ref2/x.raw $ref2/y.raw  $ref2/z.raw linear_system/rhs.raw
	# eval_nodal_function.py "x*0 + 1" $ref2/x.raw $ref2/y.raw  $ref2/z.raw linear_system/rhs.raw

	mkdir -p le_linear_system
	assemble3 $ref2 le_linear_system
fi

echo "-----------------"
echo "Laplacian"
echo "-----------------"

spmv 	1 0 linear_system linear_system/rhs.raw test.raw
cuspmv 	1 0 linear_system linear_system/rhs.raw test.raw

lapl_matrix_free $ref2 1 linear_system/rhs.raw mf_test.raw
SFEM_USE_MACRO=1 lapl_matrix_free p2 1 linear_system/rhs.raw macro_test.raw
SFEM_USE_MACRO=0 lapl_matrix_free p2 1 linear_system/rhs.raw p2_test.raw

echo "-----------------"
echo "Linear Elasticity"
echo "-----------------"

cuspmv 	1 0 le_linear_system "gen:ones" le_test.raw

linear_elasticity_matrix_free $ref2 1 "gen:ones"  le_mf_test.raw
# SFEM_USE_MACRO=1 linear_elasticity_matrix_free p2 1 "gen:ones" le_macro_test.raw
SFEM_USE_MACRO=0 linear_elasticity_matrix_free p2 1 "gen:ones" le_p2_test.raw

# Output
echo "-----------------"
echo "Output"
echo "-----------------"

mkdir -p out_p2
aos_to_soa le_p2_test.raw 8 3 out_p2/le_p2_test
# aos_to_soa le_mf_test.raw 8 3 out_p2/le_mf_test

raw_to_db.py p2 le_out.vtk --point_data="out_p2/le_p2_test.0.raw,out_p2/le_p2_test.1.raw,out_p2/le_p2_test.2.raw"

# raw_to_db.py $ref2 mf_out.vtk --point_data="linear_system/rhs.raw,mf_test.raw,macro_test.raw,test.raw,p2_test.raw"


