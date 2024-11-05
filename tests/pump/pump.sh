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

mesh=LeakageTest_Fluid

if [[ -d "$mesh" ]]
then
	echo "Reusing mesh $mesh!"
else
	# $SCRIPTPATH/../../data/exodus/LeakageTest_Fluid60K.sh
	# $SCRIPTPATH/../../data/exodus/LeakageTest_Fluid_578K.sh
	$SCRIPTPATH/../../data/exodus/LeakageTest_Fluid_4624K.sh
fi

sinlet=$mesh/sidesets_aos/FLUID_INLET.raw
soutlet1=$mesh/sidesets_aos/FLUID_OUTLET1.raw
soutlet2=$mesh/sidesets_aos/FLUID_OUTLET2.raw
swalls=$mesh/sidesets_aos/FLUID_WALLS.raw

export SFEM_DIRICHLET_NODESET="$sinlet"
export SFEM_DIRICHLET_VALUE="1"
export SFEM_DIRICHLET_COMPONENT="0"

export SFEM_CONTACT_NODESET="$soutlet1,$soutlet2"
export SFEM_CONTACT_VALUE="-10,-2"
export SFEM_CONTACT_COMPONENT="0"
export SFEM_USE_ELASTICITY=0

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 

echo "Running: obstacle $mesh output"
$LAUNCH obstacle $mesh output 

raw_to_db.py $mesh output/out.vtk --point_data=output/u.raw --point_data_type="$SFEM_REAL_T"

