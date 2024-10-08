#!/usr/bin/env bash

set -e

if [[ -z $SFEM_DIR ]]
then
	echo "SFEM_DIR must be defined with the installation prefix of sfem"
	exit 1
fi

if (($# != 1))
then
	printf "usage: $0 <mesh>\n" 1>&2
	exit -1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH
export PATH=$SFEM_DIR/bin:$PATH
export PATH=$SFEM_DIR/scripts/sfem/mesh:$PATH
export PATH=$SFEM_DIR/scripts/sfem/utils:$PATH
export PYTHONPATH=$SFEM_DIR/lib:$SFEM_DIR/scripts:$PYTHONPATH

mesh=$1

rm -rf output
mkdir -p output

count_element_to_node_incidence $mesh output/count.int32.raw   
fp_convert.py output/count.int32.raw output/count.float64.raw int32 float64
raw_to_db.py $mesh output/count.vtk -p output/count.float64.raw
