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

create_cylinder.sh 3

MESH_DIR=./mesh/surface
SYSTEM_DIR=system

mkdir -p $SYSTEM_DIR
assemble_adjaciency_matrix $MESH_DIR $SYSTEM_DIR

rm -rf eigs

N=300
graph_analysis.py $SYSTEM_DIR $N | tee log.txt
num_vectors=`grep num_vectors log.txt | awk '{print $2}'`

# raw_to_db.py $MESH_DIR x.xmf --transient --point_data='eigs/real*.raw,eigs/imag*.raw' --n_time_steps=$num_vectors
raw_to_db.py $MESH_DIR x.xmf --transient --point_data='eigs/real*.raw' --n_time_steps=$num_vectors
