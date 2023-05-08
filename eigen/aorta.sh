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

CASE_DIR=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better

MESH_DIR=mesh/surface
SYSTEM_DIR=system

mkdir -p $MESH_DIR
skin $CASE_DIR $MESH_DIR

export SFEM_GRAPH_LAPLACIAN=1
export EIG_WHICH='SR'

# export EIG_WHICH='LR'

N=20
gsp.sh $MESH_DIR $N