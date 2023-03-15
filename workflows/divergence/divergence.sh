#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH

if (($# != 4))
then
	printf "usage: $0 <mesh_path> <vx.raw> <vy.raw> <vz.raw>\n" 1>&2
	exit -1
fi

# Volume mesh
mesh_path=$1

# Vector field
vx=$2
vy=$3
vz=$4

real_type_size=8

mkdir -p workspace
workspace="workspace"

mkdir -p output
output=output

#####################################
# Surface mesh
#####################################

surf_mesh_path=$workspace/skinned

mkdir -p $surf_mesh_path


skin $mesh_path $surf_mesh_path

set -x

surface_nodes=$surf_mesh_path/node_mapping.raw
parent_elements=$surf_mesh_path/parent.raw

#####################################


volume_divergence()
{
	if (($# != 4))
	then
		printf "usage: $0 <name> <velx.raw> <vely.raw> <velz.raw>\n" 1>&2
		exit -1
	fi

	name=$1
	ux=$2
	uy=$3
	uz=$4

	divu=$workspace/div_vel.raw

	divergence $mesh_path $ux $uy $uz $divu
	div_measure=`python3 -c "import numpy as np; print(np.sum((np.fromfile(\"$divu\")), dtype=np.float64))"`
	
	lumped_mass_inv $mesh_path $divu $divu
	raw_to_db.py $mesh_path $output/"$name".vtk --point_data="$divu"

	echo "---------------------------"
	echo "[$name]: sum(div(u)) = $div_measure"
	echo "---------------------------"
}

surface_divergence()
{
	name=$1
	ux=$2
	uy=$3
	uz=$4

	divu=$workspace/div_vel.raw
	surface_outflux $surf_mesh_path $ux $uy $uz $divu
}

svx=$workspace/svx.raw
svy=$workspace/svy.raw
svz=$workspace/svz.raw

sgather $surface_nodes $real_type_size $vx $svx
sgather $surface_nodes $real_type_size $vy $svy
sgather $surface_nodes $real_type_size $vz $svz

surface_divergence "surfdiv" $svx $svy $svz

raw_to_db.py $surf_mesh_path $output/normal.vtk --cell_data="normal*.raw" --cell_data_type="float32"
