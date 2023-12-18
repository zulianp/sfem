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

# LAUNCH="mpiexec -np 8"
# LAUNCH=""

# if [[ $# -le "3" ]]
# then
# 	printf "usage: $0 <db.e> <hmax> <margin> <sdt.float32.raw> [aux_mesh]\n" 1>&2
# 	exit -1
# fi

# set -x

create_sphere.sh 3

field=field.raw
mesh=mesh
out=resampled
skinned=skinned

mkdir -p $skinned
skin $mesh $skinned

mesh_to_sdf.py $skinned $db_out --hmax=0.01 --margin=0.5

# pizzastack_to_mesh <nx> <ny> <nz> <field.raw> <mesh_folder> [output_path=./mesh_field.raw]
# pizzastack_to_mesh $nx $ny $nz $field $mesh $out
# raw_to_db.py $mesh out.vtk --point_data=$out
