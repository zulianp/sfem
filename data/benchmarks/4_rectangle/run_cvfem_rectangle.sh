#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../workflows/divergence:$PATH
PATH=$SCRIPTPATH/../../benchmarks/meshes:$PATH

export SFEM_DIRICHLET_NODES=./mesh/sidesets_aos/sleft.raw  
export SFEM_HANDLE_NEUMANN=1 
export SFEM_NEUMANN_FACES=./mesh/sidesets_aos/sright.raw 

mkdir -p system

create_box_2D.sh 5

export SFEM_HANDLE_RHS=1
export SFEM_DIRICHLET_NODES=./mesh/sidesets_aos/sleft.raw  
export SFEM_HANDLE_NEUMANN=1 
export SFEM_NEUMANN_FACES=./mesh/sidesets_aos/sright.raw
# export SFEM_NEUMANN_FACES=./mesh/sidesets_aos/stop.raw

cvfem_assemble ./mesh system
# assemble ./mesh system

UTOPIA_EXEC=$CODE_DIR/utopia/utopia/build/utopia_exec
# UTOPIA_EXEC=$CODE_DIR/utopia/utopia/build_debug/utopia_exec


if [[ -z "$UTOPIA_EXEC" ]]
then
	echo "Error! Please define UTOPIA_EXEC=<path_to_utopia_exectuable>"
	exit -1
fi

solve()
{
	mat_=$1
	rhs_=$2
	x_=$3

	echo "rhs=$rhs_"
	mpiexec -np 8 \
	$UTOPIA_EXEC -app ls_solve -A $mat_ -b $rhs_ -out $x_ -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 --verbose
}


solve system/rowptr.raw system/rhs.raw x.raw

raw_to_db.py mesh x.vtk --point_data="x.raw,system/rhs.raw"

