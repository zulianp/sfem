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

if [[ -d "hex8_mesh" ]]
then
	echo "Reusing mesh"
else
	N=150
	box_mesh.py hex8_mesh -c hex8 -x $N -y $N -z $N --height=1 --width=1 --depth=1
	box_mesh.py tet4_mesh -c tet4 -x $N -y $N -z $N --height=1 --width=1 --depth=1

	N_coarse=$(( N/2 + 1 ))
	box_mesh.py tet4_mesh_coarse -c tet4 -x $N_coarse -y $N_coarse -z $N_coarse --height=1 --width=1 --depth=1
	mesh_p1_to_p2 tet4_mesh_coarse macro_tet4_mesh
fi

laplacian_apply hex8_mesh gen:ones hex8_AxU.raw
laplacian_apply tet4_mesh gen:ones tet4_AxU.raw

SFEM_USE_MACRO=0 laplacian_apply macro_tet4_mesh gen:ones tet10_AxU.raw
SFEM_USE_MACRO=1 laplacian_apply macro_tet4_mesh gen:ones macro_tet4_AxU.raw


