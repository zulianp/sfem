#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../workflows/divergence:$PATH

set -x

nrefs=5

folder=cylinder
mesh_db=$folder/mesh.vtk
mesh_raw=./mesh
mesh_surface=$mesh_raw/surface

mkdir -p $mesh_raw
mkdir -p $mesh_surface/inlet
mkdir -p $mesh_surface/outlet
mkdir -p $mesh_surface/wall
mkdir -p $mesh_raw/sidesets_aos

mkdir -p $folder

idx_type_size=4

cylinder.py $mesh_db $nrefs
db_to_raw.py $mesh_db $mesh_raw
skin $mesh_raw $mesh_surface

select_surf $mesh_surface -1 0   0   0.99   $mesh_surface/sides_inlet.raw
select_surf $mesh_surface  1 0 	 0   0.99 	$mesh_surface/sides_outlet.raw

numbers=`mktemp`
numbers2=`mktemp`

python3 -c "import numpy as np; a=np.fromfile(\"$mesh_surface/i0.raw\", dtype=np.int32); a = np.array([i for i in range(0, len(a))], dtype=np.int32); a.tofile(\"$numbers\")"
ls -la $mesh_surface/i0.raw
ls -la $numbers


set_diff $numbers  $mesh_surface/sides_inlet.raw  $numbers2
set_diff $numbers2 $mesh_surface/sides_outlet.raw $mesh_surface/sides_wall.raw 

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

	workspace=`mktemp -d`
	# workspace=workspace
	# mkdir -p workspace


	# Convert gather surf-mesh indices
	sgather $mesh_surface/sides_"$name".raw $idx_type_size $mesh_surface/i0.raw $workspace/i0.raw
	sgather $mesh_surface/sides_"$name".raw $idx_type_size $mesh_surface/i1.raw $workspace/i1.raw
	sgather $mesh_surface/sides_"$name".raw $idx_type_size $mesh_surface/i2.raw $workspace/i2.raw


	# Convert surf-mesh indices to volume mesh indices
	sgather $workspace/i0.raw $idx_type_size $mesh_surface/node_mapping.raw $mesh_surface/"$name"/i0.raw 
	sgather $workspace/i1.raw $idx_type_size $mesh_surface/node_mapping.raw $mesh_surface/"$name"/i1.raw 
	sgather $workspace/i2.raw $idx_type_size $mesh_surface/node_mapping.raw $mesh_surface/"$name"/i2.raw 

	# set_union $mesh_surface/"$name"/i0.raw $mesh_surface/"$name"/i1.raw $workspace/temp.raw
	# set_union $workspace/temp.raw		   $mesh_surface/"$name"/i2.raw $sideset_raw

	soa_to_aos "$mesh_surface/"$name"/i*.raw" $idx_type_size $sideset_raw
	rm -r $workspace
}

boundary_nodes inlet  $mesh_raw/sidesets_aos/sinlet.raw
boundary_nodes outlet $mesh_raw/sidesets_aos/soutlet.raw
boundary_nodes wall   $mesh_raw/sidesets_aos/swall.raw

sides=$mesh_raw/dirichlet.raw
python3 -c "import numpy as np; a=np.fromfile(\"$mesh_raw/x.raw\", dtype=np.float32); a.fill(0); a.astype(np.float64).tofile(\"$sides\")"

smask $mesh_raw/sidesets_aos/sinlet.raw  $sides $sides 1
smask $mesh_raw/sidesets_aos/soutlet.raw $sides $sides 2
smask $mesh_raw/sidesets_aos/swall.raw   $sides $sides 3

raw_to_db.py $mesh_raw $mesh_raw/dirichlet.vtk --point_data="$sides"

