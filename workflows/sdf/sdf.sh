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

if (($# != 4))
then
	printf "usage: $0 <db.e> <hmax> <margin> <sdt.float32.raw>\n" 1>&2
	exit -1
fi

db_in=$1
hmax=$2
margin=$3
db_out=$4

mesh_raw=mesh_raw
skinned=skinned
surf=surf.vtk

mkdir -p $skinned
mkdir -p $mesh_raw

db_to_raw.py $db_in $mesh_raw
skin $mesh_raw $skinned


mesh_to_sdf.py $surf $hmax $margin $db_out
# raw_to_db.py $skinned $surf --point_data="nx.float32.raw,ny.float32.raw,nz.float32.raw" --point_data_type="float32,float32,float32"


mesh_to_udf.py $surf $hmax $margin $db_out
