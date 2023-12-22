#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../workflows/divergence:$PATH

LAUNCH=""
# LAUNCH=srun

if (($# != 1))
then
	printf "usage: $0 <n_refinements>\n" 1>&2
	exit -1
fi

nrefs=$1

folder=sphere
mesh_db=$folder/mesh.vtk
mesh_raw=./mesh
mesh_surface=$mesh_raw/surface

mkdir -p $mesh_raw
mkdir -p $folder

idx_type_size=4

sphere.py $mesh_db $nrefs
db_to_raw.py $mesh_db $mesh_raw -e tetra
$LAUNCH skin $mesh_raw $mesh_surface
