#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../python:$PATH
PATH=$SCRIPTPATH/../python/mesh:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH
PATH=$SCRIPTPATH/../workflows/:$PATH

source $SCRIPTPATH/../workflows/sfem_config.sh

if (($# != 1))
then
	printf "usage: $0 <mesh_folder>\n" 1>&2
	exit -1
fi


set -x

MESH_DIR=$1



# project.py "eigs/real*.raw" -1 $MESH_DIR/x.raw float32

mesh_evalf.py $MESH_DIR/x.raw $MESH_DIR/y.raw $MESH_DIR/z.raw "np.sin(3*np.pi*2*x)" f.raw
project.py "eigs/real*.raw" -1 f.raw float64

rec_vecs=(`ls reconstructed/*.raw`)
num_vectors=${#rec_vecs[@]}

raw_to_db.py $MESH_DIR r.xmf --point_data="reconstructed/rf.raw,f.raw"  --n_time_steps=$num_vectors