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

mesh=LeakageTest_Fluid

if [[ -d "$mesh" ]]
then
	echo "Reusing mesh $mesh!"
else
	$SCRIPTPATH/../../data/exodus/LeakageTest_Fluid_60K.sh
	# $SCRIPTPATH/../../data/exodus/LeakageTest_Fluid_578K.sh
	# $SCRIPTPATH/../../data/exodus/LeakageTest_Fluid_4624K.sh
fi

sinlet=$mesh/sidesets_aos/FLUID_INLET.raw
soutlet1=$mesh/sidesets_aos/FLUID_OUTLET1.raw
soutlet2=$mesh/sidesets_aos/FLUID_OUTLET2.raw
swalls=$mesh/sidesets_aos/FLUID_WALLS.raw


export SFEM_USE_ELASTICITY=1

if [[ $SFEM_USE_ELASTICITY == 1 ]]
then
	export SFEM_DIRICHLET_NODESET="$soutlet1,$soutlet1,$soutlet1"
	export SFEM_DIRICHLET_VALUE="0,0,0.01"
	export SFEM_DIRICHLET_COMPONENT="0,1,2"

	export SFEM_CONTACT_NODESET="$soutlet2"
	export SFEM_CONTACT_VALUE="0.03"
	export SFEM_CONTACT_COMPONENT="2"
else
	export SFEM_DIRICHLET_NODESET="$sinlet"
	export SFEM_DIRICHLET_VALUE="1"
	export SFEM_DIRICHLET_COMPONENT="0"

	export SFEM_CONTACT_NODESET="$soutlet1,$soutlet2"
	export SFEM_CONTACT_VALUE="-10,-2"
	export SFEM_CONTACT_COMPONENT="0"
fi


echo "Running: obstacle $mesh output"
$LAUNCH obstacle $mesh output 

if [[ $SFEM_USE_ELASTICITY == 1 ]]
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

	raw_to_db.py $mesh output/out.vtk  --point_data="output/soa/*.raw" --point_data_type="$SFEM_REAL_T"
else
	raw_to_db.py $mesh output/out.vtk --point_data=output/u.raw --point_data_type="$SFEM_REAL_T"
fi
