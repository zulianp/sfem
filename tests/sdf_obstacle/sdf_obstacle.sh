#!/usr/bin/env bash

set -e

elem_type=HEX8

N=10
box_mesh.py box -x $N -y $N -z $N --cell_type=$elem_type --tx=-0.5 --ty=-0.5 --tz=-0.5
box_mesh.py obstacle_mesh -x 2 -y 4 -z 4 --cell_type=TET4 --width=2 --height=4 --depth=4 --tx=-1 --ty=-2 --tz=-2

skin box box/skin
skin obstacle_mesh obstacle_mesh/skin

margin=0.5
hmax=0.01
mkdir -p obstacle/sdf
mesh_to_sdf.py obstacle_mesh/skin obstacle/sdf/sdf.float32.raw --hmax=$hmax --margin=$margin
raw_to_xdmf.py obstacle/sdf/sdf.float32.raw
cp obstacle/sdf/metadata_sdf.float32.yml obstacle/sdf/meta.yaml 

$LAUNCH ./sdf_obstacle.py contact_elasticity.yaml

raw_to_db.py box output/out.vtk -p 'output/*.raw'
