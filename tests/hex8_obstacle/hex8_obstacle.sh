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

if [[ -d "$mesh" ]]
then
	echo "Reusing mesh"
else
	# Cyclic mesh
	create_cyclic_ss_mesh.sh 1 $SFEM_ELEMENT_REFINE_LEVEL

	# BOX mesh for testing
	# create_box_ss_mesh.sh 8 $SFEM_ELEMENT_REFINE_LEVEL
fi

# Cyclic mesh
sinlet=$mesh/surface/sidesets_aos/inlet.raw
soutlet=$mesh/surface/sidesets_aos/outlet.raw
sobstacle=$mesh/surface/sidesets_aos/wall1.raw
./compute_distance.py mesh/viz/ $sobstacle ub.raw

# Box mesh for testing
# sinlet=$mesh/surface/sidesets_aos/left.raw 
# soutlet=$mesh/surface/sidesets_aos/right.raw 

SQP=1
export SFEM_USE_ELASTICITY=1

if [[ $SFEM_USE_ELASTICITY -eq 1 ]]
then
	if [[ $SQP -eq 1 ]]
	then
		export SFEM_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$soutlet,$soutlet,$soutlet"
		export SFEM_DIRICHLET_VALUE="0,0,0,0,0,0"
		export SFEM_DIRICHLET_COMPONENT="0,1,2,0,1,2"

		export SFEM_CONTACT_NODESET="$sobstacle"
		export SFEM_CONTACT_VALUE="path:ub.raw"
		export SFEM_CONTACT_COMPONENT="1"
	else
		export SFEM_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$soutlet,$soutlet,$soutlet"
		export SFEM_DIRICHLET_VALUE="0,0.1,0,0,-0.1,0"
		export SFEM_DIRICHLET_COMPONENT="0,1,2,0,1,2"
	fi
else
	if [[ $SQP -eq 1 ]]
	then
		# Contact
		export SFEM_DIRICHLET_NODESET="$sinlet"
		export SFEM_DIRICHLET_VALUE="1"
		export SFEM_DIRICHLET_COMPONENT="0"

		export SFEM_CONTACT_NODESET="$soutlet"
		export SFEM_CONTACT_VALUE="-1"
		export SFEM_CONTACT_COMPONENT="0"
	else
		export SFEM_DIRICHLET_NODESET="$sinlet,$soutlet"
		export SFEM_DIRICHLET_VALUE="1,-1"
		export SFEM_DIRICHLET_COMPONENT="0,0"

		rm -f output/upper_bound.raw 
	fi
fi

$LAUNCH obstacle $mesh output

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

		aos_to_soa $f 8 $dims output/soa/$name
		mv output/soa/$name".0.raw" output/soa/"$var".0."$ts".raw
		mv output/soa/$name".1.raw" output/soa/"$var".1."$ts".raw
		mv output/soa/$name".2.raw" output/soa/"$var".2."$ts".raw
	done

	raw_to_db.py $mesh/viz output/hex8.vtk  --point_data="output/soa/*.raw"
	raw_to_db.py mesh/surface/wall1/ output/obstacle.vtk --coords=mesh/viz --cell_type=quad --point_data="output/soa/upper_bound.1.*"
else
	raw_to_db.py $mesh/viz output/hex8.vtk --point_data=output/u.raw,output/rhs.raw,output/upper_bound.raw --point_data_type="$SFEM_REAL_T,$SFEM_REAL_T,$SFEM_REAL_T"
fi
