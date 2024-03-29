#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python/sfem:$PATH
PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH

if (($# != 5))
then
	printf "usage: $0 <mesh_path> <vx.raw> <vy.raw> <vz.raw> <output>\n" 1>&2
	exit -1
fi

# Volume mesh
mesh_path=$1

# Vector field
vx=$2
vy=$3
vz=$4
output=$5
div_function=u_dot_grad_q
real_type_size=8

workspace=`mktemp -d`

mkdir -p $output


#####################################
# Surface mesh
#####################################

surf_mesh_path=$workspace/skinned

mkdir -p $surf_mesh_path


skin $mesh_path $surf_mesh_path

# set -x

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
	cell_div=$workspace/cell_cdiv.raw

	echo "Volume divergence: $name"

	SFEM_VERBOSE=1 $div_function $mesh_path $ux $uy $uz $divu
	# div_measure=`python3 -c "import numpy as np; print(np.sum((np.fromfile(\"$divu\")), dtype=np.float64))"`

	integrate_divergence $mesh_path $ux $uy $uz

	cdiv $mesh_path $ux $uy $uz $cell_div

		
	cdivu=$workspace/cdiv.raw
	lumped_mass_inv $mesh_path $divu $cdivu
	raw_to_db.py $mesh_path $output/"$name".vtk --point_data="$cdivu" --cell_data="$cell_div"
}

surface_divergence()
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

	echo "-------------------------------"
	echo "Surface divergence: $name"
	echo "-------------------------------"

	outflux=$workspace/outflux.raw
	surface_outflux $surf_mesh_path $ux $uy $uz $outflux
	raw_to_db.py $surf_mesh_path $output/"$name".vtk --cell_data="$outflux"
}

volume_divergence "voldiv" $vx $vy $vz

svx=$workspace/svx.raw
svy=$workspace/svy.raw
svz=$workspace/svz.raw

sgather $surface_nodes $real_type_size $vx $svx
sgather $surface_nodes $real_type_size $vy $svy
sgather $surface_nodes $real_type_size $vz $svz

surface_divergence "surfdiv" $svx $svy $svz

# raw_to_db.py $surf_mesh_path $output/normal.vtk --cell_data="normal*.raw" --cell_data_type="float32"

# Clean-up
rm -r $workspace
