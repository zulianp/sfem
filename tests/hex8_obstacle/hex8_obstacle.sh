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

export SFEM_ELEMENT_TYPE=PROTEUS_HEX8 
export SFEM_ELEMENT_REFINE_LEVEL=8

mesh=mesh

# if [[ -d "$mesh" ]]
# then
# 	echo "Reusing mesh"
# else
	create_cyclic_ss_mesh.sh 4 $SFEM_ELEMENT_REFINE_LEVEL
	# BOX mesh for testing
	# create_box_ss_mesh.sh 20 $SFEM_ELEMENT_REFINE_LEVEL
# fi



sinlet=$mesh/surface/sidesets_aos/inlet.raw
soutlet=$mesh/surface/sidesets_aos/outlet.raw

# Box mesh for testing
# sinlet=$mesh/surface/sidesets_aos/left.raw 
# soutlet=$mesh/surface/sidesets_aos/right.raw 

export SFEM_DIRICHLET_NODESET="$sinlet,$soutlet"
export SFEM_DIRICHLET_VALUE="1,-1"
export SFEM_DIRICHLET_COMPONENT="0,0"

obstacle $mesh output

raw_to_db.py $mesh/viz $mesh/viz/hex8.vtk --point_data=output/u.raw,output/rhs.raw
