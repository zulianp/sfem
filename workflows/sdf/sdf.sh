#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../..:$PATH
export PATH=$SCRIPTPATH/../../python/sfem:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/sdf:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

LAUNCH="mpiexec -np 8"
# LAUNCH=""

if [[ $# -le "3" ]]
then
	printf "usage: $0 <db.e> <hmax> <margin> <sdt.float32.raw> [aux_mesh]\n" 1>&2
	exit -1
fi

set -x

db_in=$1
hmax=$2
margin=$3
db_out=$4
boxed_mesh_raw=boxed_mesh



if [[ -n "$5" ]]
then
	# echo "using $5 to defined bounding-box"
	opts='--scale_box=1.1 --box_from_mesh='$5
	db_to_raw.py $5 $boxed_mesh_raw
	skin $boxed_mesh_raw $boxed_mesh_raw/skinned
fi

mesh_raw=mesh_raw
skinned=skinned
surf=surf.vtk

mkdir -p $skinned
mkdir -p $mesh_raw

db_to_raw.py $db_in $mesh_raw  --select_elem_type=tetra
skin $mesh_raw $skinned
# create_dual_graph $skinned $skinned/dual

if [[ -z $REUSE_SDF ]]
then
	mesh_to_sdf.py $skinned $db_out --hmax=$hmax --margin=$margin $opts --export_normals
	raw_to_xdmf.py $db_out
	raw_to_db.py $skinned $surf --point_data="nx.float32.raw,ny.float32.raw,nz.float32.raw" --point_data_type="float32,float32,float32"
fi

if [[ -n "$5" ]]
then
	cat metadata_sdf.float32.yml | tr ':' ' ' | awk '{print $1,$2}' | tr ' ' '=' > vars.sh
	source vars.sh
	SFEM_INTERPOLATE=0 $LAUNCH gap_from_sdf $boxed_mesh_raw/skinned $nx $ny $nz $ox $oy $oz $dx $dy $dz $db_out sdf_on_mesh
	raw_to_db.py $boxed_mesh_raw/skinned gap.vtk --point_data="sdf_on_mesh/*float64.raw"

	geometry_aware_gap_from_sdf $boxed_mesh_raw/skinned $nx $ny $nz $ox $oy $oz $dx $dy $dz $db_out ga_sdf_on_mesh
	raw_to_db.py $boxed_mesh_raw/skinned ga_gap.vtk --point_data="ga_sdf_on_mesh/*float64.raw"

	extract_sharp_edges $boxed_mesh_raw/skinned 0.1 sharp_features
	cp $boxed_mesh_raw/skinned/{x,y,z}.raw sharp_features

	# Serial since the sharp features connectivity are an auxiliry graph of the actual mesh
	SFEM_INTERPOLATE=0 gap_from_sdf sharp_features $nx $ny $nz $ox $oy $oz $dx $dy $dz $db_out sdf_on_sharp
	raw_to_db.py sharp_features gap_sharp.vtk --point_data="sdf_on_sharp/*float64.raw" 
fi

