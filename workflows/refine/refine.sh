#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python/sfem:$PATH
PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

LAUNCH=""

if (($# != 2))
then
	printf "usage: $0 <db_in> <db_out>\n" 1>&2
	exit -1
fi

db_in=$1
db_out=$2
mesh_refined=./temp
mesh_raw=./temp/original

mkdir -p $mesh_refined
mkdir -p $mesh_raw

db_to_raw.py $db_in $mesh_raw
refine $mesh_raw $mesh_refined

raw_to_db.py $mesh_refined $db_out
