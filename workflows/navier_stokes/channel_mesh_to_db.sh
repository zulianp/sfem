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
PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

idx_type_size=4
folder=channel_hole
mesh_db=$folder/mesh.vtk
# mesh_db=$folder/mesh_fine.vtk
mesh_db_p2=$folder/mesh_p2.vtk
mesh_raw=./mesh
# mesh_original=./unsorted
mesh_original=$mesh_raw
mesh_surface=$mesh_raw/surface

workspace=workspace
mkdir -p $workspace


mkdir -p $mesh_original
mkdir -p $mesh_raw
mkdir -p $mesh_surface/inlet
mkdir -p $mesh_surface/outlet
mkdir -p $mesh_surface/wall
mkdir -p $mesh_raw/sidesets_aos

set -x

db_to_raw.py $mesh_db $mesh_raw/p1
mesh_p1_to_p2 $mesh_raw/p1 $mesh_raw
raw_to_db.py $mesh_raw $mesh_db_p2

export SFEM_ELEMENT_TYPE=11 #EDGE3
$LAUNCH skin $mesh_raw $mesh_surface

$LAUNCH select_surf $mesh_surface -0.1  0.2   0.2   0.99  	$mesh_surface/sides_inlet.raw
$LAUNCH select_surf $mesh_surface  2.3  0.2   0.2   0.99   	$mesh_surface/sides_outlet.raw

numbers=`mktemp`
numbers2=`mktemp`

python3 -c "import numpy as np; a=np.fromfile(\"$mesh_surface/i0.raw\", dtype=np.int32); a = np.array([i for i in range(0, len(a))], dtype=np.int32); a.tofile(\"$numbers\")"
ls -la $mesh_surface/i0.raw
ls -la $numbers


$LAUNCH set_diff $numbers  $mesh_surface/sides_inlet.raw  $numbers2
$LAUNCH set_diff $numbers2 $mesh_surface/sides_outlet.raw $mesh_surface/sides_wall.raw 

rm $numbers  $numbers2

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

	# workspace=`mktemp -d`

	for(( i=0; i < 3; i++ ))
	do
		fname=i"$i".raw
		# Convert gather surf-mesh indices
		$LAUNCH sgather $mesh_surface/sides_"$name".raw $idx_type_size $mesh_surface/$fname $workspace/$fname
	
		# Convert surf-mesh indices to volume mesh indices
		$LAUNCH sgather $workspace/$fname $idx_type_size $mesh_surface/node_mapping.raw $mesh_surface/"$name"/$fname 
	done	

	$LAUNCH soa_to_aos "$mesh_surface/"$name"/i*.raw" $idx_type_size $sideset_raw
	$LAUNCH unique $sideset_raw $sideset_raw
	# rm -r $workspace
}

boundary_nodes inlet  $mesh_raw/sidesets_aos/sinlet.raw
boundary_nodes outlet $mesh_raw/sidesets_aos/soutlet.raw
boundary_nodes wall   $mesh_raw/sidesets_aos/swall.raw

sides=$mesh_raw/dirichlet.raw
python3 -c "import numpy as np; a=np.fromfile(\"$mesh_raw/x.raw\", dtype=np.float32); a.fill(0); a.astype(np.float64).tofile(\"$sides\")"

$LAUNCH smask $mesh_raw/sidesets_aos/sinlet.raw  $sides $sides 1
$LAUNCH smask $mesh_raw/sidesets_aos/soutlet.raw $sides $sides 2
$LAUNCH smask $mesh_raw/sidesets_aos/swall.raw   $sides $sides 3

raw_to_db.py $mesh_raw $mesh_raw/dirichlet.vtk --point_data="$sides"

