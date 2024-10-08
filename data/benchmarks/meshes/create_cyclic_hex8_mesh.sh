#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

SFEM_ROOT=$SCRIPTPATH/../../../

export PATH=$SFEM_ROOT/../matrix.io:$PATH

export PATH=$SCRIPTPATH:$PATH
export PATH=$SFEM_ROOT/python/sfem:$PATH
export PATH=$SFEM_ROOT/python/sfem/mesh:$PATH
export PATH=$SFEM_ROOT/python/sfem/grid:$PATH
export PATH=$SFEM_ROOT/python/sfem/algebra:$PATH
export PATH=$SFEM_ROOT/python/sfem/utils:$PATH
export PATH=$SFEM_ROOT/data/benchmarks/meshes:$PATH

if [[ -z $SFEM_BIN_DIR ]]
then
	PATH=$SFEM_ROOT/build:$PATH
	source $SFEM_ROOT/build/sfem_config.sh
else
	echo "Using binaries in $SFEM_BIN_DIR"
	PATH=$SFEM_BIN_DIR:$PATH
	source $SFEM_BIN_DIR/sfem_config.sh
fi

if (($# != 1))
then
	printf "usage: $0 <resolution>\n" 1>&2
	exit -1
fi

resolution=$1

mesh=mesh

mkdir -p $mesh
cyclic_sector_mesh.py $resolution
db_to_raw.py model.vtk $mesh --select_elem_type=hexahedron

skin $mesh $mesh/surface

SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 1.925 -0.01 0.46 0.99 	$mesh/surface/sides_outlet.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface -1.925 -0.01 0.46 0.99 	$mesh/surface/sides_inlet.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0 1 0.5 0.95 				$mesh/surface/sides_wall0.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface -1 1.7 0.5 0.95 			$mesh/surface/sides_wall1.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0 1.4 0 0.99 				$mesh/surface/sides_symm0.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0 1.4 1 0.99 				$mesh/surface/sides_symm1.raw


boundary_nodes()
{
	if (($# != 2))
	then
		printf "usage: $0 <name> <sideset.raw>\n" 1>&2
		exit -1
	fi

	set -x

	name=$1
	sideset_raw=$2

	mkdir -p $mesh/surface/"$name"
	# workspace=`mktemp -d`
	workspace=workspace
	mkdir -p $workspace
	
	# Convert gather surf-mesh indices
	$LAUNCH sgather $mesh/surface/sides_"$name".raw $SFEM_IDX_SIZE $mesh/surface/i0.raw $workspace/i0.raw
	$LAUNCH sgather $mesh/surface/sides_"$name".raw $SFEM_IDX_SIZE $mesh/surface/i1.raw $workspace/i1.raw
	$LAUNCH sgather $mesh/surface/sides_"$name".raw $SFEM_IDX_SIZE $mesh/surface/i2.raw $workspace/i2.raw
	$LAUNCH sgather $mesh/surface/sides_"$name".raw $SFEM_IDX_SIZE $mesh/surface/i3.raw $workspace/i3.raw

	# Convert surf-mesh indices to volume mesh indices
	$LAUNCH sgather $workspace/i0.raw $SFEM_IDX_SIZE $mesh/surface/node_mapping.raw $mesh/surface/"$name"/i0.raw 
	$LAUNCH sgather $workspace/i1.raw $SFEM_IDX_SIZE $mesh/surface/node_mapping.raw $mesh/surface/"$name"/i1.raw 
	$LAUNCH sgather $workspace/i2.raw $SFEM_IDX_SIZE $mesh/surface/node_mapping.raw $mesh/surface/"$name"/i2.raw 
	$LAUNCH sgather $workspace/i3.raw $SFEM_IDX_SIZE $mesh/surface/node_mapping.raw $mesh/surface/"$name"/i3.raw 

	$LAUNCH soa_to_aos "$mesh/surface/"$name"/i*.raw" $SFEM_IDX_SIZE $sideset_raw
	# rm -r $workspace
}


mkdir -p $mesh/sidesets_aos/
boundary_nodes inlet  $mesh/sidesets_aos/sinlet.raw
boundary_nodes outlet $mesh/sidesets_aos/soutlet.raw
boundary_nodes wall1   $mesh/sidesets_aos/swall1.raw

sides=$mesh/dirichlet.raw
python3 -c "import numpy as np; a=np.fromfile(\"$mesh/surface/x.raw\", dtype=np.float32); a.fill(0); a.astype(np.float64).tofile(\"$sides\")"

# smask $mesh/sidesets_aos/wall0.raw $sides $sides 3
smask $mesh/sidesets_aos/wall1.raw $sides $sides 4
# smask $mesh/sidesets_aos/symm0.raw $sides $sides 5
# smask $mesh/sidesets_aos/symm1.raw $sides $sides 6
smask $mesh/sidesets_aos/soutlet.raw $sides $sides 1
smask $mesh/sidesets_aos/sinlet.raw $sides $sides 2

raw_to_db.py $mesh/surface $mesh/dirichlet.vtk --point_data="$sides" --cell_type=quad


