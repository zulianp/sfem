#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH

LAUNCH=""
# LAUNCH=srun

if (($# != 1))
then
	printf "usage: $0 <n_refinements>\n" 1>&2
	exit -1
fi

set -x

nrefs=$1

folder=square
mesh_db=$folder/mesh.vtk
mesh_raw=./mesh
mesh_surface=$mesh_raw/surface

mkdir -p $mesh_raw
mkdir -p $mesh_surface/left
mkdir -p $mesh_surface/right
mkdir -p $mesh_surface/top
mkdir -p $mesh_surface/bottom
mkdir -p $mesh_raw/sidesets_aos

mkdir -p $folder

idx_type_size=4

box_2D.py $mesh_db $nrefs 1 1
db_to_raw.py $mesh_db $mesh_raw

# set -x

$LAUNCH skin $mesh_raw $mesh_surface
raw_to_db.py $mesh_surface $mesh_surface/surf.vtk

$LAUNCH select_surf $mesh_surface -1  0.5  0  0.99 $mesh_surface/sides_left.raw
$LAUNCH select_surf $mesh_surface  1  0.5  0  0.99 $mesh_surface/sides_right.raw
$LAUNCH select_surf $mesh_surface  0.5 -1  0  0.99 $mesh_surface/sides_bottom.raw
$LAUNCH select_surf $mesh_surface  0.5  1  0  0.99 $mesh_surface/sides_top.raw

print_array()
{
	if (($# != 1))
	then
		printf "usage: $0 <input.raw>\n" 1>&2
		exit -1
	fi

	input=$1

	python3 -c "import numpy as np; a=np.fromfile(\"$input\", dtype=np.int32); print(a);"
}

boundary_nodes()
{
	if (($# != 2))
	then
		printf "usage: $0 <name> <sideset.raw>\n" 1>&2
		exit -1
	fi

	name=$1
	sideset_raw=$2

	workspace=`mktemp -d`
	
	# Convert gather surf-mesh indices
	$LAUNCH sgather $mesh_surface/sides_"$name".raw $idx_type_size $mesh_surface/i0.raw $workspace/i0.raw
	$LAUNCH sgather $mesh_surface/sides_"$name".raw $idx_type_size $mesh_surface/i1.raw $workspace/i1.raw

	# Convert surf-mesh indices to volume mesh indices
	$LAUNCH sgather $workspace/i0.raw $idx_type_size $mesh_surface/node_mapping.raw $mesh_surface/"$name"/i0.raw 
	$LAUNCH sgather $workspace/i1.raw $idx_type_size $mesh_surface/node_mapping.raw $mesh_surface/"$name"/i1.raw 

	$LAUNCH soa_to_aos "$mesh_surface/"$name"/i*.raw" $idx_type_size $sideset_raw
	rm -r $workspace
}

boundary_nodes left  	$mesh_raw/sidesets_aos/sleft.raw
boundary_nodes right 	$mesh_raw/sidesets_aos/sright.raw
boundary_nodes bottom   $mesh_raw/sidesets_aos/sbottom.raw
boundary_nodes top   	$mesh_raw/sidesets_aos/stop.raw

sides=$mesh_raw/dirichlet.raw
python3 -c "import numpy as np; a=np.fromfile(\"$mesh_raw/x.raw\", dtype=np.float32); a.fill(0); a.astype(np.float64).tofile(\"$sides\")"

$LAUNCH smask $mesh_raw/sidesets_aos/sleft.raw    $sides $sides 1
$LAUNCH smask $mesh_raw/sidesets_aos/sright.raw   $sides $sides 2
$LAUNCH smask $mesh_raw/sidesets_aos/sbottom.raw  $sides $sides 3
$LAUNCH smask $mesh_raw/sidesets_aos/stop.raw  	  $sides $sides 4

raw_to_db.py $mesh_raw $mesh_raw/dirichlet.vtk --point_data="$sides"

