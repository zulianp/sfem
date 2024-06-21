#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../build:$PATH
PATH=$SCRIPTPATH/../python/sfem:$PATH
PATH=$SCRIPTPATH/../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH
PATH=$SCRIPTPATH/../../matrix.io:$PATH

HERE=$PWD

mkdir -p le_test
cd le_test

mkdir -p output
# export OMP_NUM_THREADS=12
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 

# rm -rf mesh
# create_cylinder.sh 0

# create_cylinder_p2.sh 4
# export SFEM_USE_MACRO=1

sleft=mesh/sidesets_aos/sinlet.raw
sright=mesh/sidesets_aos/soutlet.raw

set -x

if [[ "$1" == "LE" ]]
then
	export SFEM_OPERATOR="LinearElasticity"
	export VAR_UX=0
	export VAR_UY=1
	export VAR_UZ=2
	export SFEM_BLOCK_SIZE=3

	export SFEM_DIRICHLET_NODESET="$sleft,$sleft,$sleft,$sright,$sright,$sright"
	export SFEM_DIRICHLET_VALUE="-0.2,0.05,0,0.2,0,-0.05"
	export SFEM_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ"
else
	export SFEM_OPERATOR="Laplacian"
	export SFEM_BLOCK_SIZE=1

	export SFEM_DIRICHLET_NODESET="$sleft,$sright"
	export SFEM_DIRICHLET_VALUE="-1,1"
	export SFEM_DIRICHLET_COMPONENT="0,0"
fi

export SFEM_USE_GPU=1
export SFEM_USE_PRECONDITIONER=1
export CUDA_LAUNCH_BLOCKING=0

# $LAUNCH steady_state_sim mesh output
$LAUNCH mgsolve mesh output

if [[ $SFEM_BLOCK_SIZE != 1 ]]
then
	aos_to_soa output/x.raw 8 $SFEM_BLOCK_SIZE output/disp
	aos_to_soa output/rhs.raw 8 $SFEM_BLOCK_SIZE output/rhs
	aos_to_soa output/r.raw 8 $SFEM_BLOCK_SIZE output/r

	raw_to_db.py mesh output/x.vtk -p "output/disp.*.raw,output/rhs.*.raw,output/r.*.raw"
else
	raw_to_db.py mesh output/x.vtk -p "output/x.raw,output/c*.raw,output/residual*.raw"
fi

cd $HERE
