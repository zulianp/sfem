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

create_sphere.sh 1

MESH_DIR=mesh/surface
SYSTEM_DIR=system

# export SFEM_GRAPH_LAPLACIAN=1
# export EIG_WHICH='SM'
# export SFEM_REMOVE_LOWER_TRIANGULAR=1

# export EIG_WHICH='LR'
# export EIG_WHICH='LM'

export SFEM_DIRECTED=0

# N=638
N=200
USE_DENSE=1 \
gsp.sh $MESH_DIR $N