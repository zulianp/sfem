#!/usr/bin/env bash

set -e

if [[ -z $SFEM_DIR ]]
then
	echo "SFEM_DIR must be defined with the installation prefix of sfem"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH
export PATH=$SFEM_DIR/bin:$PATH
export PATH=$SFEM_DIR/scripts/sfem/mesh:$PATH
export PYTHONPATH=$SFEM_DIR/lib:$SFEM_DIR/scripts:$PYTHONPATH
source $SFEM_DIR/workflows/sfem_config.sh

mesh=mesh
N=10
box_mesh.py $mesh -c tetra -x $N -y $N -z $N --height=1 --width=1 --depth=1


# OP=LinearElasticity
OP=Laplacian

./amg_test.py $mesh $OP