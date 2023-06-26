#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../python/algebra:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

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
	$UTOPIA_EXEC -app ls_solve -A $mat_ -b $rhs_ -out $x_ -use_amg false --use_ksp -pc_type lu -ksp_type preonly
}

mesh=mesh
# create_square.sh 0
# rm $mesh/z.raw
nvars=2

dirichlet_nodes=all.raw
top=$mesh/sidesets_aos/stop.raw 
bottom=$mesh/sidesets_aos/sbottom.raw 

cat $top 	>  $dirichlet_nodes
cat $bottom >> $dirichlet_nodes

neumman_system=neumman_system
block_system=block_system
system=system

rm -rf $neumman_system
rm -rf $block_system
rm -rf $system


export SFEM_MU=1
export SFEM_LAMBDA=1

set -x
linear_elasticity_assemble $mesh $neumman_system
crs_apply_dirichlet $nvars $nvars $neumman_system $dirichlet_nodes 1 $block_system

# for ((i = 0; i < $nvars; i++))
# do
# 	smask $top $neumman_system/rhs."$i".raw 
# done

smask $top $neumman_system/rhs.1.raw $block_system/rhs.1.raw 0.1
smask $bottom $block_system/rhs.1.raw $block_system/rhs.1.raw "-0.1"
cp $neumman_system/rhs.0.raw $block_system/rhs.0.raw 

# stokes $mesh stokes_block_system
crs.py $nvars $nvars $block_system $system
blocks.py $block_system'/rhs.*.raw' $system/rhs.raw

solve $system/rowptr.raw $system/rhs.raw x.raw

unblocks.py $nvars x.raw
unblocks.py $nvars $system/rhs.raw

raw_to_db.py $mesh out.vtk --point_data="x.*.raw,system/rhs.*.raw"
