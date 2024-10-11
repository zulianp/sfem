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
export SFEM_ELEMENT_REFINE_LEVEL=8

mesh=mesh

if [[ -d "$mesh" ]]
then
	echo "Reusing mesh"
else
	create_box_ss_mesh.sh 35 $SFEM_ELEMENT_REFINE_LEVEL
fi

# Box mesh for testing
sinlet=$mesh/surface/sidesets_aos/left.raw 
soutlet=$mesh/surface/sidesets_aos/right.raw 

export SFEM_USE_ELASTICITY=0

if [[ $SFEM_USE_ELASTICITY -eq 1 ]]
then
	export SFEM_BLOCK_SIZE=3
	export SFEM_OPERATOR="LinearElasticity"
	export SFEM_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$soutlet,$soutlet,$soutlet"
	export SFEM_DIRICHLET_VALUE="0,0.1,0,0,-0.1,0"
	export SFEM_DIRICHLET_COMPONENT="0,1,2,0,1,2"
else
	export SFEM_BLOCK_SIZE=1
	export SFEM_OPERATOR="Laplacian"
	export SFEM_DIRICHLET_NODESET="$sinlet,$soutlet"
	export SFEM_DIRICHLET_VALUE="1,-1"
	export SFEM_DIRICHLET_COMPONENT="0,0"
fi

export SFEM_HEX8_ASSUME_AFFINE=1
export SFEM_MATRIX_FREE=1
export SFEM_COARSE_MATRIX_FREE=1

export SFEM_USE_CRS_GRAPH_RESTRICT=0
export SFEM_CRS_MEM_CONSERVATIVE=1

export SFEM_USE_CHEB=1
export SFEM_CHEB_EIG_MAX_SCALE=1
export SFEM_CHEB_EIG_TOL=1e-5
export SFEM_SMOOTHER_SWEEPS=20

export SFEM_MAX_IT=20
export SFEM_MG=1
export SFEM_USE_PRECONDITIONER=0

export SFEM_VERBOSITY_LEVEL=1
export SFEM_DEBUG=0

# export SFEM_COARSE_TOL=1e-12

$LAUNCH mgsolve $mesh output

if [[ $SFEM_USE_ELASTICITY -eq 1 ]]
then
	dims=3
	files=`ls output/*.raw`
	mkdir -p output/soa

	for f in ${files[@]}
	do
		name=`basename $f`
		var=`echo $name | tr '.' ' ' | awk '{print $1}'`
		ts=`echo $name  | tr '.' ' ' | awk '{print $2}'`

		aos_to_soa $f $SFEM_REAL_SIZE $dims output/soa/$name
		mv output/soa/$name".0.raw" output/soa/"$var".0."$ts".raw
		mv output/soa/$name".1.raw" output/soa/"$var".1."$ts".raw
		mv output/soa/$name".2.raw" output/soa/"$var".2."$ts".raw
	done

	raw_to_db.py $mesh/viz output/out.vtk  --point_data="output/soa/*.raw" --point_data_type="$SFEM_REAL_T"
else
	raw_to_db.py $mesh/viz output/out.vtk --point_data=output/x.raw,output/rhs.raw --point_data_type="$SFEM_REAL_T,$SFEM_REAL_T"
fi
