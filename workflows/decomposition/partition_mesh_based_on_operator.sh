#!/usr/bin/env bash

source ../sfem_config.sh

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH

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

partition_mesh_based_on_operator $mesh $matrix $num_partitions $output
