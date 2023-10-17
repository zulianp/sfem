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

mkdir -p vtk_meshes

o=0.1
d=0.8
box.py vtk_meshes/small_cube.vtk $o $o $o $d $d $d
box.py vtk_meshes/cube.vtk 0 0 0 1 1 1

small_cube_raw=small_cube_raw
cube_raw=cube_raw

db_to_raw.py vtk_meshes/small_cube.vtk 	$small_cube_raw tetra
db_to_raw.py vtk_meshes/cube.vtk 		$cube_raw 		tetra

skin $small_cube_raw  	$small_cube_raw/surface
skin $cube_raw  		$cube_raw/surface

hmax=0.01
margin=0.01
sdf=./sdf.float32.raw
# opts='--scale_box=1 --box_from_mesh='$small_cube_raw
mesh_to_sdf.py $cube_raw/surface $sdf --hmax=$hmax --margin=$margin $opts
raw_to_xdmf.py $sdf

cat metadata_sdf.float32.yml | tr ':' ' ' | awk '{print $1,$2}' | tr ' ' '=' > vars.sh
source vars.sh
SFEM_INTERPOLATE=0 gap_from_sdf $small_cube_raw/surface $nx $ny $nz $ox $oy $oz $dx $dy $dz $sdf sdf_on_mesh
raw_to_db.py $small_cube_raw/surface gap.vtk --point_data="sdf_on_mesh/*.float64.raw"
