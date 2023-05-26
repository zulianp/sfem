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

mesh_evalf.py $MESH_DIR/x.raw $MESH_DIR/y.raw $MESH_DIR/z.raw "2*np.sin(2*np.pi*2*x) + np.sin(3*np.pi*2*x) + 0.3*np.sin(40*np.pi*2*x) + 0.4*np.sin(6.1*np.pi*2*x)" f.raw


# mesh_evalf.py $MESH_DIR/x.raw $MESH_DIR/y.raw $MESH_DIR/z.raw "y*y" f.raw
# project.py "eigs/real*.raw" -1 0.1 f.raw float64
project.py "eigs/real*.raw" -1 0 f.raw float64

rec_vecs=(`ls reconstructed/*.raw`)
num_vectors=${#rec_vecs[@]}

raw_to_db.py $MESH_DIR r.vtk --point_data="reconstructed/rf.raw,reconstructed/if.raw,f.raw" 
