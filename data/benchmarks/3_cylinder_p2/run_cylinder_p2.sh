#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../workflows/divergence:$PATH


UTOPIA_EXEC=$CODE_DIR/utopia/utopia/build/utopia_exec

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

# mesh_path=./mesh
mesh_path=../2_darcy_cylinder/mesh
# workspace=`mktemp -d`
workspace=workspace
mkdir -p $workspace
mkdir -p $workspace/system

export SFEM_HANDLE_DIRICHLET=1
export SFEM_HANDLE_NEUMANN=1
export SFEM_HANDLE_RHS=1

export SFEM_NEUMANN_FACES=$mesh_path/sidesets_aos/soutlet.raw
export SFEM_DIRICHLET_NODES=$mesh_path/sidesets_aos/sinlet.raw

# lldb -- 
assemble $mesh_path $workspace/system

solve $workspace/system/rowptr.raw $workspace/system/rhs.raw $workspace/x.raw

raw_to_db.py $mesh_path x.vtk --point_data="$workspace/x.raw,$workspace/system/rhs.raw"


# Clean-up
# rm -r $workspace
