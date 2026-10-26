#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PATH=$SCRIPTPATH/../../:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../python/sfem:$PATH
PATH=$SCRIPTPATH/../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH
PATH=$SCRIPTPATH/../../matrix.io:$PATH

if [[ -z $SFEM_BIN_DIR ]]
then
	PATH=$SCRIPTPATH/../../build:$PATH
	source $SCRIPTPATH/../../build/sfem_config.sh
else
	echo "Using binaries in $SFEM_BIN_DIR"
	PATH=$SFEM_BIN_DIR:$PATH
	source $SFEM_BIN_DIR/sfem_config.sh
fi

HERE=$PWD

mkdir -p le_test
cd le_test

mkdir -p output

export OMP_NUM_THREADS=8
# export OMP_NUM_THREADS=288
# export OMP_NUM_THREADS=12
export OMP_PROC_BIND=true 

# export SFEM_MESH_REFINE=0
# create_cylinder_p2.sh 3
export SFEM_USE_MACRO=1

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
export CUDA_LAUNCH_BLOCKING=0

$LAUNCH mgsolve mesh output

if [[ $SFEM_BLOCK_SIZE != 1 ]]
then
	aos_to_soa output/x.raw $SFEM_REAL_SIZE $SFEM_BLOCK_SIZE output/disp
	aos_to_soa output/rhs.raw $SFEM_REAL_SIZE $SFEM_BLOCK_SIZE output/rhs
	raw_to_db.py mesh output/x.vtk -p "output/disp.*.raw,output/rhs.*.raw" -d "$SFEM_REAL_T,$SFEM_REAL_T"
else
	raw_to_db.py mesh output/x.vtk -p "output/x.raw,output/c*.raw,output/residual*.raw" -d "$SFEM_REAL_T,$SFEM_REAL_T,$SFEM_REAL_T"
fi

cd $HERE
