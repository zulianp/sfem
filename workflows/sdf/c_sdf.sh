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

if [[ $# -le "4" ]]
then
	printf "usage: $0 <db.e> <hmax> <margin> <sdt.float32.raw> [aux_mesh]\n" 1>&2
	exit -1
fi

set -x

db_in=$1
hmax=$2
margin=$3
db_out=$4
boxed_mesh_raw=boxed_mesh

if [[ -n "$5" ]]
then
	db_to_raw.py $5 $boxed_mesh_raw
	skin $boxed_mesh_raw $boxed_mesh_raw/skinned
	export SFEM_BOXED_MESH=$boxed_mesh_raw
	export SFEM_SCALE_BOX="1.1"
fi

mesh_raw=mesh_raw
skinned=skinned
surf=surf.vtk

mkdir -p $skinned
mkdir -p $mesh_raw

db_to_raw.py $db_in $mesh_raw
skin $mesh_raw $skinned
# <mesh> <hmax> <margin> <output_folder>
# lldb -- 

OMP_NUM_THREADS=8 OMP_PROC_BIND=true mesh_to_sdf $skinned $hmax $margin $db_out 
raw_to_xdmf.py $db_out/sdf.float32.raw

raw_to_db.py $skinned $surf --point_data="nx.float32.raw,ny.float32.raw,nz.float32.raw" --point_data_type="float32,float32,float32"


if [[ -n "$5" ]]
then
	cat $db_out/metadata_sdf.float32.yml | tr ':' ' ' | awk '{print $1,$2}' | tr ' ' '=' > vars.sh
	source vars.sh
	SFEM_INTERPOLATE=1 $LAUNCH gap_from_sdf $boxed_mesh_raw/skinned $nx $ny $nz $ox $oy $oz $dx $dy $dz $db_out/sdf.float32.raw sdf_on_mesh
	raw_to_db.py $boxed_mesh_raw/skinned gap.vtk --point_data="sdf_on_mesh/*float64.raw"
fi


