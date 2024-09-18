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

if (($# != 2))
then
	printf "usage: $0 <resolution> <tassellation_level>\n" 1>&2
	exit -1
fi

resolution=$1

export SFEM_ELEMENT_TYPE=PROTEUS_HEX8 
export SFEM_ELEMENT_REFINE_LEVEL=$2

mesh=mesh

mkdir -p $mesh

cyclic_sector_mesh.py $resolution
db_to_raw.py model.vtk $mesh --select_elem_type=hexahedron

skin $mesh $mesh/macro_quad_surface
cp $mesh/{x,y,z}.raw $mesh/macro_quad_surface

proteus_quad4_to_quad4.py $SFEM_ELEMENT_REFINE_LEVEL $mesh/macro_quad_surface $mesh/surface

# Generate full-mesh for visualization
proteus_hex8_to_hex8 $SFEM_ELEMENT_REFINE_LEVEL $mesh $mesh/viz

SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 1.4 -0.01 0.46 0.99 	$mesh/surface/sides_outlet.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface -1.4 -0.01 0.46 0.99 	$mesh/surface/sides_inlet.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0 1 0.5 0.95 			$mesh/surface/sides_wall0.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface -1 1.7 0.5 0.95 		$mesh/surface/sides_wall1.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0 1.4 0 0.99 			$mesh/surface/sides_symm0.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0 1.4 1 0.99 			$mesh/surface/sides_symm1.raw

boundary_nodes()
{
	if (($# != 3))
	then
		printf "usage: $0 <mesh> <name> <sideset.raw>\n" 1>&2
		exit -1
	fi
	set -x

	local mesh=$1
	local name=$2
	local sideset_raw=$3

	local workspace=`mktemp -d`

	for(( i=0; i < 4; i++ ))
	do
		local fname=i"$i".raw

		mkdir -p $mesh/"$name"
		# Convert gather surf-mesh indices
		$LAUNCH sgather $mesh/sides_"$name".raw $SFEM_IDX_SIZE $mesh/$fname $mesh/"$name"/$fname
	done	

	$LAUNCH soa_to_aos "$mesh/"$name"/i*.raw" $SFEM_IDX_SIZE $sideset_raw
	rm -r $workspace
}

mkdir -p $mesh/surface/sidesets_aos/
boundary_nodes $mesh/surface outlet $mesh/surface/sidesets_aos/outlet.raw
boundary_nodes $mesh/surface inlet  $mesh/surface/sidesets_aos/inlet.raw
boundary_nodes $mesh/surface wall0  $mesh/surface/sidesets_aos/wall0.raw
boundary_nodes $mesh/surface wall1  $mesh/surface/sidesets_aos/wall1.raw
boundary_nodes $mesh/surface symm0  $mesh/surface/sidesets_aos/symm0.raw
boundary_nodes $mesh/surface symm1  $mesh/surface/sidesets_aos/symm1.raw

sides=$mesh/dirichlet.raw
python3 -c "import numpy as np; a=np.fromfile(\"$mesh/surface/x.raw\", dtype=np.float32); a.fill(0); a.astype(np.float64).tofile(\"$sides\")"

smask $mesh/surface/sidesets_aos/wall0.raw $sides $sides 3
smask $mesh/surface/sidesets_aos/wall1.raw $sides $sides 4
smask $mesh/surface/sidesets_aos/symm0.raw $sides $sides 5
smask $mesh/surface/sidesets_aos/symm1.raw $sides $sides 6
smask $mesh/surface/sidesets_aos/outlet.raw $sides $sides 1
smask $mesh/surface/sidesets_aos/inlet.raw $sides $sides 2

raw_to_db.py $mesh/surface $mesh/dirichlet.vtk --point_data="$sides" --cell_type=quad


