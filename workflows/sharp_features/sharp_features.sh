#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../python/grid:$PATH
PATH=$SCRIPTPATH/../../python/algebra:$PATH
PATH=$SCRIPTPATH/../../python/sdf:$PATH
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

extract_sharp_edges $skinned 0.1 sharp_features
cp $skinned/{x,y,z}.raw sharp_features
raw_to_db.py sharp_features sharp_edges.vtk

cp $skinned/{x,y,z}.raw sharp_features/corners
raw_to_db.py sharp_features/corners corners.vtk
