#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../python:$PATH
PATH=$SCRIPTPATH/../python/mesh:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH

set -x

create_box_2D.sh 1

MESH_DIR=mesh
SYSTEM_DIR=system

mkdir -p $SYSTEM_DIR
assemble_adjaciency_matrix $MESH_DIR $SYSTEM_DIR

rm -rf eigs

N=4
graph_analysis.py $SYSTEM_DIR $N
raw_to_db.py $MESH_DIR x.xmf --transient --point_data='eigs/real*.raw' --n_time_steps=$N
