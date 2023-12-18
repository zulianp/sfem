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


create_sphere.sh 2

field=field.raw
mesh=mesh
out=resampled
skinned=skinned
sdf=sdf.float32.raw

mkdir -p $skinned
skin $mesh $skinned

mesh_to_sdf.py $skinned $sdf --hmax=0.01 --margin=0.5
raw_to_xdmf.py $sdf

sizes=`head -3 metadata_sdf.float32.yml 			  | awk '{print $2}' | tr '\n' ' '`
origins=`head -8 metadata_sdf.float32.yml 	| tail -3 | awk '{print $2}' | tr '\n' ' '`
scaling=`head -11 metadata_sdf.float32.yml 	| tail -3 | awk '{print $2}' | tr '\n' ' '`

echo $sizes
echo $origins
echo $scaling

# SFEM_INTERPOLATE=1 
SFEM_READ_FP32=1 pizzastack_to_mesh $sizes $origins $scaling $sdf $mesh $field
raw_to_db.py $mesh out.vtk --point_data=$field
