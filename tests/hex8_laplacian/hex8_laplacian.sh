#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


export PATH=$SCRIPTPATH/../../:$PATH
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


if [[ -d "hex8_mesh" ]]
then
	echo "Reusing mesh"
else
	N=100
	box_mesh.py hex8_mesh -c hex8 -x $N -y $N -z $N --height=1 --width=1 --depth=1
	box_mesh.py tet4_mesh -c tet4 -x $N -y $N -z $N --height=1 --width=1 --depth=1
fi

laplacian_apply hex8_mesh gen:ones hex8_AxU.raw
laplacian_apply tet4_mesh gen:ones tet4_AxU.raw
