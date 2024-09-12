#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../../matrix.io:$PATH

export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../..:$PATH
export PATH=$SCRIPTPATH/../../python/sfem:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/utils:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

if [[ -z $SFEM_BIN_DIR ]]
then
	PATH=$SCRIPTPATH/../../build:$PATH
	source $SCRIPTPATH/../../build/sfem_config.sh
else
	echo "Using binaries in $SFEM_BIN_DIR"
	PATH=$SFEM_BIN_DIR:$PATH
	source $SFEM_BIN_DIR/sfem_config.sh
fi

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 
export CUDA_LAUNCH_BLOCKING=0

if [[ -d "hex8_mesh" ]]
then
	echo "Reusing mesh"
else
	N=100
	box_mesh.py hex8_mesh -c hex8 -x $N -y $N -z $N --height=1 --width=1 --depth=1
fi

laplacian_apply hex8_mesh gen:ones hex8_AxU.raw

sleft=hex8_mesh/boundary_nodes/left.int32.raw
sright=hex8_mesh/boundary_nodes/right.int32.raw

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

export SFEM_USE_GPU=0
export SFEM_USE_PRECONDITIONER=0
export SFEM_MATRIX_FREE=1
export SFEM_HEX8_ASSUME_AFFINE=1
export SFEM_HEX8_ASSUME_AXIS_ALIGNED=1
export SFEM_USE_CHEB=0

$LAUNCH mgsolve hex8_mesh output

if [[ $SFEM_BLOCK_SIZE != 1 ]]
then
	aos_to_soa output/x.raw 8 $SFEM_BLOCK_SIZE output/disp
	aos_to_soa output/rhs.raw 8 $SFEM_BLOCK_SIZE output/rhs
	# aos_to_soa output/r.raw 8 $SFEM_BLOCK_SIZE output/r

	raw_to_db.py hex8_mesh output/x.vtk -p "output/disp.*.raw,output/rhs.*.raw"
else
	raw_to_db.py hex8_mesh output/x.vtk -p "output/x.raw,output/c*.raw,output/residual*.raw"
fi
