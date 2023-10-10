#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../python/algebra:$PATH
PATH=$SCRIPTPATH/../../python/sdf:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

LAUNCH=""

if (($# -le "4"))
then
	printf "usage: $0 <db.e> <hmax> <margin> <sdt.float32.raw> [aux_mesh]\n" 1>&2
	exit -1
fi

set -x

db_in=$1
hmax=$2
margin=$3
db_out=$4

if !((-z $5))
then
	opts='--scale_box=1.1 --box_from_mesh='$5
fi

mesh_raw=mesh_raw
skinned=skinned
surf=surf.vtk

mkdir -p $skinned
mkdir -p $mesh_raw

db_to_raw.py $db_in $mesh_raw
skin $mesh_raw $skinned
# create_dual_graph $skinned $skinned/dual
mesh_to_sdf.py $skinned $db_out --hmax=$hmax --margin=$margin $opts

raw_to_db.py $skinned $surf --point_data="nx.float32.raw,ny.float32.raw,nz.float32.raw" --point_data_type="float32,float32,float32"

