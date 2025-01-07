#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


export PATH=$SCRIPTPATH/../../../matrix.io:$PATH

export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../../python/sfem:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/utils:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

if [[ -z $SFEM_BIN_DIR ]]
then
	PATH=$SCRIPTPATH/../../build:$PATH
	source $SCRIPTPATH/../../build/sfem_config.sh
else
	echo "Using binaries in $SFEM_BIN_DIR"
	PATH=$SFEM_BIN_DIR:$PATH
	source $SFEM_BIN_DIR/sfem_config.sh
fi

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 
export CUDA_LAUNCH_BLOCKING=0

idx_type_size=4

# NX=5
# NY=5
# NZ=5
# box_mesh.py hex8_mesh_2 -c hex8 -x $NX -y $NY -z $NZ --height=1 --width=1 --depth=1


cyclic_sector_mesh.py
db_to_raw.py model.vtk hex8_mesh_2 --select_elem_type=hexahedron

export SFEM_ELEMENT_TYPE=HEX8
skin hex8_mesh_2 hex8_mesh_surface_std

export SFEM_ELEMENT_TYPE=SSHEX8 
export SFEM_ELEMENT_REFINE_LEVEL=8

skin hex8_mesh_2 hex8_mesh_surface
cp hex8_mesh_2/{x,y,z}.raw ./hex8_mesh_surface

ssquad4_to_quad4.py $SFEM_ELEMENT_REFINE_LEVEL hex8_mesh_surface hex8_mesh_surface/quad4
raw_to_db.py hex8_mesh_surface/quad4 quad4_mesh.vtk --cell_type=quad
raw_to_db.py hex8_mesh_surface_std quad4_mesh_std.vtk --cell_type=quad



SFEM_ELEMENT_TYPE=QUAD4 select_surf hex8_mesh_surface/quad4 1.4 -0.01 0.46 0.99 hex8_mesh_surface/quad4/sides_outlet.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf hex8_mesh_surface/quad4 -1.4 -0.01 0.46 0.99 hex8_mesh_surface/quad4/sides_inlet.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf hex8_mesh_surface/quad4 0 1 0.5 0.95 hex8_mesh_surface/quad4/sides_wall0.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf hex8_mesh_surface/quad4 -1 1.7 0.5 0.95 hex8_mesh_surface/quad4/sides_wall1.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf hex8_mesh_surface/quad4 0 1.4 0 0.99 hex8_mesh_surface/quad4/sides_symm0.raw
SFEM_ELEMENT_TYPE=QUAD4 select_surf hex8_mesh_surface/quad4 0 1.4 1 0.99 hex8_mesh_surface/quad4/sides_symm1.raw


# SFEM_ELEMENT_TYPE=QUAD4 select_surf hex8_mesh_surface_std  	1.5  0. 0.5 0.8 	hex8_mesh_surface_std/sides_outlet.raw



boundary_nodes()
{
	if (($# != 3))
	then
		printf "usage: $0 <mesh> <name> <sideset.raw>\n" 1>&2
		exit -1
	fi
	set -x

	mesh=$1
	name=$2
	sideset_raw=$3

	mkdir -p workspace
	workspace=workspace

	# workspace=`mktemp -d`

	for(( i=0; i < 4; i++ ))
	do
		fname=i"$i".raw

		mkdir -p $mesh/"$name"
		# Convert gather surf-mesh indices
		$LAUNCH sgather $mesh/sides_"$name".raw $idx_type_size $mesh/$fname $mesh/"$name"/$fname
	
		# Convert surf-mesh indices to volume mesh indices
		# $LAUNCH sgather $workspace/$fname $idx_type_size $mesh/node_mapping.raw $mesh/"$name"/$fname 
	done	

	$LAUNCH soa_to_aos "$mesh/"$name"/i*.raw" $idx_type_size $sideset_raw
	# rm -r $workspace
}

mkdir -p hex8_mesh_surface/quad4/sidesets_aos/
boundary_nodes hex8_mesh_surface/quad4/ outlet  hex8_mesh_surface/quad4/sidesets_aos/outlet.raw
boundary_nodes hex8_mesh_surface/quad4/ inlet  hex8_mesh_surface/quad4/sidesets_aos/inlet.raw
boundary_nodes hex8_mesh_surface/quad4/ wall0  hex8_mesh_surface/quad4/sidesets_aos/wall0.raw
boundary_nodes hex8_mesh_surface/quad4/ wall1  hex8_mesh_surface/quad4/sidesets_aos/wall1.raw
boundary_nodes hex8_mesh_surface/quad4/ symm0  hex8_mesh_surface/quad4/sidesets_aos/symm0.raw
boundary_nodes hex8_mesh_surface/quad4/ symm1  hex8_mesh_surface/quad4/sidesets_aos/symm1.raw


sides=hex8_mesh_surface/quad4/dirichlet.raw
python3 -c "import numpy as np; a=np.fromfile(\"hex8_mesh_surface/quad4/x.raw\", dtype=np.float32); a.fill(0); a.astype(np.float64).tofile(\"$sides\")"

smask hex8_mesh_surface/quad4/sidesets_aos/wall0.raw $sides $sides 3
smask hex8_mesh_surface/quad4/sidesets_aos/wall1.raw $sides $sides 4
smask hex8_mesh_surface/quad4/sidesets_aos/symm0.raw $sides $sides 5
smask hex8_mesh_surface/quad4/sidesets_aos/symm1.raw $sides $sides 6
smask hex8_mesh_surface/quad4/sidesets_aos/outlet.raw $sides $sides 1
smask hex8_mesh_surface/quad4/sidesets_aos/inlet.raw $sides $sides 2

raw_to_db.py hex8_mesh_surface/quad4 dirichlet.vtk --point_data="$sides" --cell_type=quad

# raw_to_db.py hex8_mesh_surface/quad4/sides_outlet.raw sinline.vtk --cell_type=quad
