#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../python:$PATH
PATH=$SCRIPTPATH/../python/mesh:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH

if (($# != 2))
then
	printf "usage: $0 <mesh_folder> <n_vectors>\n" 1>&2
	exit -1
fi

set -x

MESH_DIR=$1
N=$2

SYSTEM_DIR=system

if [[ -z "$EIG_WHICH" ]]
then
	export EIG_WHICH='LR'
fi

mkdir -p $SYSTEM_DIR
assemble_adjaciency_matrix $MESH_DIR $SYSTEM_DIR

rm -rf eigs

graph_analysis.py $SYSTEM_DIR $EIG_WHICH $N | tee log.txt
num_vectors=`grep num_vectors log.txt | awk '{print $2}'`

# raw_to_db.py $MESH_DIR x.xmf --transient --point_data='eigs/real*.raw,eigs/imag*.raw' --n_time_steps=$num_vectors
raw_to_db.py $MESH_DIR x.xmf --transient --point_data='eigs/real*.raw' --n_time_steps=$num_vectors
raw_to_db.py $MESH_DIR dbg.xmf --point_data='count.raw'
