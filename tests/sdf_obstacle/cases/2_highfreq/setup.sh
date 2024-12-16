#!/usr/bin/env bash

set -e

elem_type=HEX8

which box_mesh.py
which highfreq_surface.py      

N=20
box_mesh.py mesh -x $N -y $N -z $N --cell_type=$elem_type --tx=0.5 --ty=0.5 --tz=-1
highfreq_surface.py obstacle_mesh -x 120 -y 120 --width=2 --height=2

raw_to_db.py obstacle_mesh obs.vtk
raw_to_db.py mesh mesh.vtk

# raw_to_db.py mesh/surface/bottom bottom.vtk --coords=mesh --cell_type=quad
# raw_to_db.py mesh/surface/top top.vtk --coords=mesh --cell_type=quad
# raw_to_db.py mesh/surface/back back.vtk --coords=mesh --cell_type=quad
# raw_to_db.py mesh/surface/front front.vtk --coords=mesh --cell_type=quad

skin mesh mesh/skin
raw_to_db.py mesh/skin skin.vtk --cell_type=quad

mkdir -p obstacle
echo "# Generated by setup.sh" > obstacle/meta.yaml
# echo "surface: ../mesh/surface/left" >> obstacle/meta.yaml
echo "surface: ../mesh/skin" >> obstacle/meta.yaml
echo "sdf: sdf" >> obstacle/meta.yaml
echo "rpath: true" >> obstacle/meta.yaml
echo "variational: true" >> obstacle/meta.yaml

# To skip expensive SDF generation uncomment following line
# exit 0

margin=1
# hmax=0.01
hmax=0.02
mkdir -p obstacle/sdf
mesh_to_sdf.py obstacle_mesh obstacle/sdf/sdf.float32.raw --hmax=$hmax --margin=$margin
raw_to_xdmf.py obstacle/sdf/sdf.float32.raw
cp obstacle/sdf/metadata_sdf.float32.yml obstacle/sdf/meta.yaml 

