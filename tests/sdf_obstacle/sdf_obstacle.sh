#!/usr/bin/env bash

set -e

elem_type=HEX8

box_mesh.py box -x 10 -y 9 -z 8 --cell_type=$elem_type
box_mesh.py obstacle_mesh -x 2 -y 2 -z 2 --cell_type=TET4 --height=2 --width=2 --depth=2

skin box box/skin
skin obstacle_mesh obstacle_mesh/skin

margin=0.5
hmax=0.008

mkdir -p obstacle/sdf
mesh_to_sdf.py obstacle_mesh/skin obstacle/sdf/sdf.float32.raw --hmax=$hmax --margin=$margin
raw_to_xdmf.py obstacle/sdf/sdf.float32.raw
cp obstacle/sdf/metadata_sdf.float32.yml obstacle/sdf/meta.yaml 

./sdf_obstacle.py contact_elasticity.yaml

raw_to_db.py box output/out.vtk -p output/g.0.raw
