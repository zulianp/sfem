#!/usr/bin/env bash

set -e


if [[ -z $SFEM_DIR ]]
then
	echo "SFEM_DIR must be defined with the installation prefix of sfem"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH
export PATH=$SFEM_DIR/bin:$PATH
export PATH=$SFEM_DIR/scripts/sfem/mesh:$PATH
export PYTHONPATH=$SFEM_DIR/lib:$SFEM_DIR/scripts:$PYTHONPATH
source $SFEM_DIR/workflows/sfem_config.sh

if (($# != 1))
then
	printf "usage: $0 <tassellation_level>\n" 1>&2
	exit -1
fi

if [[ $1 -lt 2 ]]
then
	echo "tassellation_level must be larger than 1"
	exit -1
fi

export SFEM_ELEMENT_TYPE=SSHEX8 
export SFEM_ELEMENT_REFINE_LEVEL=$1

mesh=joint_hex_db

mkdir -p $mesh
mkdir -p $mesh/viz
mkdir -p $mesh/macro_quad_surface

rm -rf $mesh/surface

db_to_raw.py $SCRIPTPATH/joint-hex.vtk $mesh/original --select_elem_type=hexahedron

if [[ $SFEM_REFINE -ge 1 ]]
then
	echo "Refined mesh!"
	sshex8_to_hex8 $(( 1 + $SFEM_REFINE )) $mesh/original $mesh/refined
	sfc $mesh/refined $mesh
else
	sfc $mesh/original $mesh
fi

hex8_fix_ordering $mesh $mesh
skin $mesh $mesh/macro_quad_surface
cp $mesh/{x,y,z}.raw $mesh/macro_quad_surface

ssquad4_to_quad4.py $SFEM_ELEMENT_REFINE_LEVEL $mesh/macro_quad_surface $mesh/surface

sshex8_to_hex8 $SFEM_ELEMENT_REFINE_LEVEL $mesh $mesh/viz

SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface -0.41 	0.095 	-0.094 	0.99 $mesh/surface/sides_base.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0.426 	0.248 	-0.222 	0.90 $mesh/surface/sides_top.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0.07 		-0.136 	0.43  	0.99 $mesh/surface/sides_top_small.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface -0.16 	0.0319  -0.315  0.95 $mesh/surface/sides_inner.raw


boundary_nodes()
{
	if (($# != 3))
	then
		printf "usage: $0 <mesh> <name> <sideset.raw>\n" 1>&2
		exit -1
	fi

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
boundary_nodes $mesh/surface base 		$mesh/surface/sidesets_aos/base.raw
boundary_nodes $mesh/surface top 		$mesh/surface/sidesets_aos/top.raw
boundary_nodes $mesh/surface top_small  $mesh/surface/sidesets_aos/top_small.raw
boundary_nodes $mesh/surface inner  	$mesh/surface/sidesets_aos/inner.raw

if [[ $SFEM_DEBUG == 1 ]]
then
	sides=$mesh/dirichlet.raw
	python3 -c "import numpy as np; a=np.fromfile(\"$mesh/surface/x.raw\", dtype=np.float32); a.fill(0); a.astype(np.float64).tofile(\"$sides\")"

	smask $mesh/surface/sidesets_aos/base.raw $sides $sides 1
	smask $mesh/surface/sidesets_aos/top.raw $sides $sides 2
	smask $mesh/surface/sidesets_aos/top_small.raw $sides $sides 3
	smask $mesh/surface/sidesets_aos/inner.raw $sides $sides 4
	raw_to_db.py $mesh/surface $mesh/viz/dirichlet.vtk --point_data="$sides" --cell_type=quad
	raw_to_db.py $mesh/viz $mesh/viz/mesh.vtk
fi
