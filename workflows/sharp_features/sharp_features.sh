#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python/sfem:$PATH
PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
PATH=$SCRIPTPATH/../../python/sfem/sdf:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

LAUNCH="mpiexec -np 8"
# LAUNCH=""

if [[ $# -le "1" ]]
then
	printf "usage: $0 <db.e> <out_raw>\n" 1>&2
	exit -1
fi

set -x

db_in=$1
db_out=$2

workspace=workspace
mesh_raw=workspace/mesh
skinned=workspace/skinned

mkdir -p $workspace
mkdir -p $skinned
mkdir -p $mesh_raw

db_to_raw.py $db_in $mesh_raw
skin $mesh_raw $skinned

extract_sharp_edges $skinned 0.15 sharp_features
cp $skinned/{x,y,z}.raw sharp_features
raw_to_db.py sharp_features sharp_edges.vtk

cp $skinned/{x,y,z}.raw sharp_features/corners
raw_to_db.py sharp_features/corners corners.vtk

cp $skinned/{x,y,z}.raw sharp_features/disconnected
raw_to_db.py sharp_features/disconnected disconnected.vtk
