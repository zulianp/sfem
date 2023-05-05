#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../python:$PATH
PATH=$SCRIPTPATH/../python/mesh:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH

set -x

clean_workspace.sh

create_sphere.sh 4

MESH_DIR=surf
SYSTEM_DIR=system

skin mesh $MESH_DIR


mkdir -p $SYSTEM_DIR
assemble_adjaciency_matrix $MESH_DIR $SYSTEM_DIR

rm -rf eigs

N=60
graph_analysis.py $SYSTEM_DIR $N | tee log.txt
num_vectors=`grep num_vectors log.txt | awk '{print $2}'`

raw_to_db.py $MESH_DIR x.xmf --transient --point_data='eigs/real*.raw' --n_time_steps=$num_vectors
