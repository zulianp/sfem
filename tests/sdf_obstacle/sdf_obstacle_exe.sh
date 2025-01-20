#!/usr/bin/env bash

set -e

if (($# != 1))
then
	echo "usage $0 <case_dir>"
	exit 1
fi

echo "SFEM_DIR=$SFEM_DIR"

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 
export CUDA_LAUNCH_BLOCKING=0
# export SFEM_ELEMENT_TYPE=SSHEX8 
export SFEM_MAX_IT=10
export SFEM_CONTACT_LINEARIZATIONS=10
# export SFEM_USE_STEEPEST_DESCENT=1
export SFEM_HEX8_ASSUME_AFFINE=1
export SFEM_MG_ENABLE_COARSE_SPACE_PRECONDITIONER=1

# nodeset=mesh/boundary_nodes/back.int32.raw
# export SFEM_DIRICHLET_NODESET="$nodeset,$nodeset,$nodeset" 

sideset=mesh/surface/sidesets/back 
export SFEM_DIRICHLET_SIDESET="$sideset,$sideset,$sideset" 

export SFEM_DIRICHLET_VALUE="0,0,0.6"
export SFEM_DIRICHLET_COMPONENT="0,1,2"
export SFEM_CONTACT_CONDITIONS=obstacle
export SFEM_DAMPING=1

export SFEM_FINE_OP_TYPE=MF


HERE=$PWD
CASE_DIR=$1
cd $CASE_DIR

ls $sideset

# export SFEM_ELEMENT_REFINE_LEVEL=`grep "refine_level" input.yaml | awk '{print $2}'`
export SFEM_ELEMENT_REFINE_LEVEL=8
echo "SFEM_ELEMENT_REFINE_LEVEL=$SFEM_ELEMENT_REFINE_LEVEL"

export SFEM_HEX8_ASSUME_AFFINE=1
# export SFEM_FIRST_LAME_PARAMETER=3.333
# export SFEM_SHEAR_MODULUS=0.357

young_modulus=1
poisson_ratio=0.46

mu=`python3 -c	'young_modulus='$young_modulus';  poisson_ratio='$poisson_ratio'; print(young_modulus / (2 * (1 + poisson_ratio)))'`
lambda=`python3 -c 'poisson_ratio='$poisson_ratio'; mu='$mu'; print(2.0 * (mu * poisson_ratio) / (1 - 2 * poisson_ratio ))'`

export SFEM_FIRST_LAME_PARAMETER=$lambda
export SFEM_SHEAR_MODULUS=$mu



which sdf_obstacle
$LAUNCH sdf_obstacle mesh output

cd -

if [[ $SFEM_ELEMENT_REFINE_LEVEL -ge 2 ]]
then
	raw_to_db.py $CASE_DIR/output/ssmesh out.vtk -p "$CASE_DIR/output/*.raw" #--point_data_type=float32
else
	raw_to_db.py $CASE_DIR/mesh out.vtk 		 -p "$CASE_DIR/output/*.raw" #--point_data_type=float32
fi
# raw_to_db.py $CASE_DIR/mesh/viz out.vtk -p "$CASE_DIR/output/*.raw"
