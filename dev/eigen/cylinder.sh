#!/usr/bin/env bash

./clean_workspace.sh

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../python:$PATH
PATH=$SCRIPTPATH/../python/mesh:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH

set -x

create_cylinder.sh 2

MESH_DIR=mesh/surface
SYSTEM_DIR=system

# export SFEM_GRAPH_LAPLACIAN=1
# export EIG_WHICH='SR'

export USE_DENSE=1
export EIG_WHICH='LR'
# export EIG_WHICH='LM'

export SFEM_DIRECTED=1
# export SFEM_NEGATE_LOWER_TRIANGULAR=1

# N=100
N=3538
gsp.sh $MESH_DIR $N
