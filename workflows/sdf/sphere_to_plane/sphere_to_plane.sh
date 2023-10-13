#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../python/grid:$PATH
PATH=$SCRIPTPATH/../../../python/algebra:$PATH
PATH=$SCRIPTPATH/../../../python/sdf:$PATH
PATH=$SCRIPTPATH/../../../data/benchmarks/meshes:$PATH

LAUNCH=""

set -x

create_sphere.sh 1
boxed_mesh_raw=sphere_mesh
mv ./mesh $boxed_mesh_raw

mesh_raw=plane_mesh
hmax=0.1
margin=0.1
sdf=sdf.float32.raw


opts='--scale_box=1.1 --box_from_mesh='$boxed_mesh_raw
mesh_to_sdf.py $mesh_raw/surface $sdf --hmax=$hmax --margin=$margin $opts
raw_to_xdmf.py $sdf

cat metadata_sdf.float32.yml | tr ':' ' ' | awk '{print $1,$2}' | tr ' ' '=' > vars.sh
source vars.sh
gap_from_sdf $boxed_mesh_raw/surface $nx $ny $nz $ox $oy $oz $dx $dy $dz $sdf sdf_on_mesh
raw_to_db.py $boxed_mesh_raw/surface gap.vtk --point_data="sdf_on_mesh/*float64.raw"
