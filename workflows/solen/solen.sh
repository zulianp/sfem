#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

mesh_raw="mesh"
mkdir -p $mesh_raw
mkdir -p $mesh_raw/skin

# mesh_path=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better

cylinder.py cylinder.vtk 6
mesh_path="$mesh_raw/p1"
mkdir -p $mesh_path
db_to_raw.py cylinder.vtk $mesh_path


# 1) Convert p1 mesh to p2 mesh
mesh_p1_to_p2 $mesh_path $mesh_raw

# raw_to_db.py $mesh_raw mesh.vtk

# 2) Skin mesh
skin $mesh_raw $mesh_raw/skin

raw_to_db.py $mesh_raw/skin surf.vtk