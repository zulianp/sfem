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

export SFEM_ELEMENT_TYPE=PROTEUS_HEX8 
export SFEM_ELEMENT_REFINE_LEVEL=$1

mesh=LeakageTest_Fluid
mkdir -p $mesh
exodusII_to_raw.py $SCRIPTPATH/LeakageTest_Fluid_60K.exo $mesh

sidesets=(`ls $mesh/sidesets`)
aos_folder=$mesh/sidesets_aos

mkdir -p $aos_folder

for ss in ${sidesets[@]}
do
	name=`basename $ss`
	echo "> $name"
	soa_to_aos "$mesh/sidesets/$ss/$name.*.raw" $SFEM_IDX_SIZE $aos_folder/"$name".raw
done
