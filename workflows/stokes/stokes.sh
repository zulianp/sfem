#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../python/algebra:$PATH

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

mesh=$SCRIPTPATH/../../data/benchmarks/4_rectangle/mesh
SFEM_DIRICHLET_NODES=$mesh/sidesets_aos/all.raw
nvars=3

stokes $mesh stokes_block_system
crs.py $nvars $nvars stokes_block_system stokes_system

blocks.py 'stokes_block_system/rhs.*.raw' stokes_system/rhs.raw

solve stokes_system/rowptr.raw stokes_system/rhs.raw x.raw
