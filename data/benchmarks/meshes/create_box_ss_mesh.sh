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
N=$(( resolution * 2 ))

export SFEM_ELEMENT_TYPE=SSHEX8 
export SFEM_ELEMENT_REFINE_LEVEL=$2

if [[ -z $SFEM_BOX_SIZE ]]
then
	SFEM_BOX_SIZE=1
else
	echo "SFEM_BOX_SIZE=$SFEM_BOX_SIZE"
fi

mesh=mesh

mkdir -p $mesh

box_mesh.py $mesh -c hex8 -x $N -y $N -z $N --skip_boundary=1
raw_to_db.py $mesh model.vtk

skin $mesh $mesh/macro_quad_surface
cp $mesh/{x,y,z}.raw $mesh/macro_quad_surface

ssquad4_to_quad4.py $SFEM_ELEMENT_REFINE_LEVEL $mesh/macro_quad_surface $mesh/surface

SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0 0.5 0.5 0.99 	$mesh/surface/sides_left.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 1 0.5 0.5 0.99 	$mesh/surface/sides_right.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0.5 0 0.5 0.99	$mesh/surface/sides_bottom.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0.5 1 0.5 0.99	$mesh/surface/sides_top.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0.5 0.5 0 0.99 	$mesh/surface/sides_back.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf $mesh/surface 0.5 0.5 1 0.99 	$mesh/surface/sides_front.raw


boundary_nodes()
{
	if (($# != 3))
	then
		printf "usage: $0 <mesh> <name> <sideset.raw>\n" 1>&2
		exit -1
	fi
	# set -x

	local mesh=$1
	local name=$2
	local sideset_raw=$3

	local workspace=`mktemp -d`

	mkdir -p $mesh/"$name"

	echo "# Automatically generated by $0" > $mesh/$name/meta.yaml
	echo "elem_num_nodes: 4" >>  $mesh/$name/meta.yaml
	echo "element_type: QUADSHELL4" >>  $mesh/$name/meta.yaml
	echo "elements:" >>  $mesh/$name/meta.yaml

	for(( i=0; i < 4; i++ ))
	do
		local fname=i"$i".raw

		# Convert gather surf-mesh indices
		$LAUNCH sgather $mesh/sides_"$name".raw $SFEM_IDX_SIZE $mesh/$fname $mesh/"$name"/$fname

		echo "$LAUNCH sgather $mesh/sides_"$name".raw $SFEM_IDX_SIZE $mesh/$fname $mesh/"$name"/$fname"

		echo "- i0: $fname" >>  $mesh/$name/meta.yaml
	done	

	$LAUNCH soa_to_aos "$mesh/"$name"/i[0-3].raw" $SFEM_IDX_SIZE $sideset_raw

	echo "points: parent" >>  $mesh/$name/meta.yaml
	echo "rpath: true" >>  $mesh/$name/meta.yaml

	# od -i $sideset_raw
	rm -r $workspace
}

mkdir -p $mesh/surface/sidesets_aos/
boundary_nodes $mesh/surface left 	$mesh/surface/sidesets_aos/left.raw
boundary_nodes $mesh/surface right  $mesh/surface/sidesets_aos/right.raw
boundary_nodes $mesh/surface bottom $mesh/surface/sidesets_aos/bottom.raw
boundary_nodes $mesh/surface top  	$mesh/surface/sidesets_aos/top.raw
boundary_nodes $mesh/surface front  $mesh/surface/sidesets_aos/front.raw
boundary_nodes $mesh/surface back  	$mesh/surface/sidesets_aos/back.raw

# sides=$mesh/dirichlet.raw
# python3 -c "import numpy as np; a=np.fromfile(\"$mesh/surface/x.raw\", dtype=np.float32); a.fill(0); a.astype(np.float64).tofile(\"$sides\")"

# smask $mesh/surface/sidesets_aos/left.raw $sides $sides 1
# smask $mesh/surface/sidesets_aos/right.raw $sides $sides 2
# smask $mesh/surface/sidesets_aos/bottom.raw $sides $sides 3
# smask $mesh/surface/sidesets_aos/top.raw $sides $sides 4
# smask $mesh/surface/sidesets_aos/front.raw $sides $sides 5
# smask $mesh/surface/sidesets_aos/back.raw $sides $sides 6

box_mesh.py $mesh -c hex8 -x $N -y $N -z $N --height=$SFEM_BOX_SIZE --width=$SFEM_BOX_SIZE --depth=$SFEM_BOX_SIZE --skip_boundary=1 $BOX_MESH_EXTRA_OPTIONS 
ssquad4_to_quad4.py $SFEM_ELEMENT_REFINE_LEVEL $mesh/macro_quad_surface $mesh/surface

# raw_to_db.py $mesh/surface $mesh/dirichlet.vtk --point_data="$sides" --cell_type=quad


