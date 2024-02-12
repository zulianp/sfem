#!/usr/bin/env bash

set -e


SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python/sfem:$PATH


if (($# != 4))
then
	printf "usage: $0 <mesh_folder> <matrix_folder> <num_partitions> <output.raw>\n" 1>&2
	exit -1
fi

set -x

mesh=$1
matrix=$2
num_partitions=$3
output=$4

export DYLD_LIBRARY_PATH=$METIS_DIR/lib
partition_mesh_based_on_operator $mesh $matrix $num_partitions $output
