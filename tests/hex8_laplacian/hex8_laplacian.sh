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
	PATH=$SFEM_BIN_DIR:$PATH
	source $SFEM_BIN_DIR/sfem_config.sh
fi

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 

box_mesh.py mesh -c hex8 -x 3 -y 3 -z 3 --height=1 --width=1 --depth=1

SFEM_USE_MACRO=0 laplacian_apply mesh gen:ones AxU.raw
