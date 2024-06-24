#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../:$PATH
export PATH=$SCRIPTPATH/../build/:$PATH
export PATH=$SCRIPTPATH/../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../python/sfem:$PATH
PATH=$SCRIPTPATH/../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH
PATH=$SCRIPTPATH/../../matrix.io:$PATH

if [[ $# -lt 1 ]]
then
	printf "usage: $0 <mesh>\n" 1>&2
fi

mesh=$1
mkdir -p output
export OMP_NUM_THREADS=4 
export OMP_PROC_BIND=true 


./test_linear_elasticity.py $mesh output
aos_to_soa output/out.raw 8 3 output/out
raw_to_db.py $mesh output/x.vtk -p "output/out.*.raw"

