#!/usr/bin/env bash

set -e
set -x

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH


raw_to_db()
{
	mesh_folder=$1
	temp_dir=`mktemp -d`

	fp_convert.py $mesh_folder/parts/part_00000/node_mapping.raw 	$temp_dir/node.raw int32 float32 
	fp_convert.py $mesh_folder/parts/part_00000/node_owner.raw 		$temp_dir/owner.raw int32 float32 
	raw_to_db.py  $mesh_folder/parts/part_00000 					$mesh_folder/part_0.vtk  \
	 	--point_data="$temp_dir/owner.raw,$temp_dir/node.raw" --point_data_type="float32,float32"

	fp_convert.py $mesh_folder/parts/part_00001/node_mapping.raw 	$temp_dir/node.raw int32 float32 
	fp_convert.py $mesh_folder/parts/part_00001/node_owner.raw 		$temp_dir/owner.raw int32 float32 
	raw_to_db.py  $mesh_folder/parts/part_00001 					$mesh_folder/part_1.vtk  \
		--point_data="$temp_dir/owner.raw,$temp_dir/node.raw" --point_data_type="float32,float32"

	rm -r $temp_dir
}

# makemesh.py

workspace=workspace

mkdir -p $workspace

case=../../data/benchmarks/2_darcy_cylinder/mesh
# case=meshes/3
p1_mesh=$workspace/p1
p2_mesh=$workspace/p2 

sfc $case $p1_mesh
mesh_p1_to_p2 $p1_mesh $p2_mesh



mpiexec -np 2 partition $p1_mesh $p1_mesh/parts

sfc $p2_mesh $p2_mesh/sorted 
mpiexec -np 2 partition $p2_mesh/sorted  $p2_mesh/parts

# mpiexec -np 2 partition $p2_mesh $p2_mesh/parts

raw_to_db $p1_mesh
raw_to_db $p2_mesh

