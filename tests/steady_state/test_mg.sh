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

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 
export CUDA_LAUNCH_BLOCKING=0
export SFEM_ELEMENT_REFINE_LEVEL=4

mesh=mesh

if [[ -d "$mesh" ]]
then
	echo "Reusing mesh"
else
	export SFEM_BOX_SIZE=1
	create_box_ss_mesh.sh 5 $SFEM_ELEMENT_REFINE_LEVEL
fi

export SFEM_BLOCK_SIZE=3
export SFEM_OPERATOR="LinearElasticity"

# export SFEM_BLOCK_SIZE=1
# export SFEM_OPERATOR="Laplacian"


# export SFEM_SHEAR_MODULUS=1
# export SFEM_FIRST_LAME_PARAMETER=1
export SFEM_HEX8_ASSUME_AFFINE=1
export SFEM_HEX8_ASSUME_AXIS_ALIGNED=0
$LAUNCH test_galerkin_assembly $mesh output | tee test_mg_log.txt

# raw_to_db.py $mesh error.vtk -p output/error.raw -d $SFEM_REAL_T
