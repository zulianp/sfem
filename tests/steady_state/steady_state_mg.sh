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
export SFEM_ELEMENT_REFINE_LEVEL=6

CASE=3

# Generate mesh db
case $CASE in
	1 | 2)
		# BOX mesh
		mesh=mesh

		if [[ -d "$mesh" ]]
		then
			echo "Reusing mesh"
		else
			create_box_ss_mesh.sh 40 $SFEM_ELEMENT_REFINE_LEVEL
		fi

		sinlet=$mesh/surface/sidesets_aos/left.raw 
		soutlet=$mesh/surface/sidesets_aos/right.raw 
	;;
	3 | 4)
		# JOINT mesh
		mesh=joint_hex_db
		if [[ -d "$mesh" ]]
		then
			echo "Reusing mesh"
		else
			# export SFEM_REFINE=4
			$SCRIPTPATH/../../data/vtk/joint-hex.sh $SFEM_ELEMENT_REFINE_LEVEL
		fi
		sinlet=$mesh/surface/sidesets_aos/base.raw
		soutlet=$mesh/surface/sidesets_aos/top_small.raw
	;;
	*)
		echo "Wrong case number $CASE"
		exit 1
	;;
esac

# Set-up BVP
case $CASE in
	1)
		export SFEM_USE_ELASTICITY=1
		export SFEM_BLOCK_SIZE=3
		export SFEM_OPERATOR="LinearElasticity"
		export SFEM_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$soutlet,$soutlet,$soutlet"
		export SFEM_DIRICHLET_VALUE="0,0.1,0,0,-0.1,0"
		export SFEM_DIRICHLET_COMPONENT="0,1,2,0,1,2"
	;;
	2)
		export SFEM_USE_ELASTICITY=0
		export SFEM_BLOCK_SIZE=1
		export SFEM_OPERATOR="Laplacian"
		export SFEM_DIRICHLET_NODESET="$sinlet,$soutlet"
		export SFEM_DIRICHLET_VALUE="1,-1"
		export SFEM_DIRICHLET_COMPONENT="0,0"
	;;
	3)
		export SFEM_USE_ELASTICITY=1
		export SFEM_BLOCK_SIZE=3
		export SFEM_OPERATOR="LinearElasticity"
		export SFEM_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$soutlet,$soutlet,$soutlet"
		export SFEM_DIRICHLET_VALUE="0,0,0,-0.1,0,0.1"
		export SFEM_DIRICHLET_COMPONENT="0,1,2,0,1,2"
	;;
	4)
		export SFEM_USE_ELASTICITY=0
		export SFEM_BLOCK_SIZE=1
		export SFEM_OPERATOR="Laplacian"
		export SFEM_DIRICHLET_NODESET="$sinlet,$soutlet"
		export SFEM_DIRICHLET_VALUE="1,0"
		export SFEM_DIRICHLET_COMPONENT="0,0"
	;;
	*)
		echo "Wrong case number $CASE"
		exit 1
	;;
esac

# Parametrize solver
export SFEM_MG=1
export SFEM_USE_CHEB=$SFEM_MG
export SFEM_MAX_IT=60
# export SFEM_MAX_IT=4000

export SFEM_HEX8_ASSUME_AFFINE=0
export SFEM_MATRIX_FREE=1
export SFEM_COARSE_MATRIX_FREE=1
export SFEM_COARSE_TOL=1e-12

export SFEM_USE_CRS_GRAPH_RESTRICT=0
export SFEM_CRS_MEM_CONSERVATIVE=1

export SFEM_CHEB_EIG_MAX_SCALE=1.0001
export SFEM_CHEB_EIG_TOL=1e-3
export SFEM_SMOOTHER_SWEEPS=10

export SFEM_USE_PRECONDITIONER=0

export SFEM_VERBOSITY_LEVEL=1
export SFEM_HEX8_QUADRATURE_ORDER=1
# export SFEM_DEBUG=1

$LAUNCH mgsolve $mesh output 
# | tee log.txt

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
	raw_to_db.py $mesh/viz output/out.vtk --point_data="output/x.raw,output/rhs.raw" --point_data_type="$SFEM_REAL_T,$SFEM_REAL_T"
fi
