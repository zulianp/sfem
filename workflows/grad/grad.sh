#!/usr/bin/env bash

source ../sfem_config.sh

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH

if (($# != 3))
then
	printf "usage: $0 <mesh_path> <f.raw> <output_folder>\n" 1>&2
	exit -1
fi

# Volume mesh
mesh_path=$1
f=$2
output_folder=$3

mkdir -p $output_folder

dfdx=$output_folder/gradx.raw
dfdy=$output_folder/grady.raw
dfdz=$output_folder/gradz.raw

grad_and_project $mesh_path $f $dfdx $dfdy $dfdz
raw_to_db.py $mesh_path $output_folder/grad.vtk --point_data="$f,$dfdx,$dfdy,$dfdz" --point_data_type="$py_sfem_real_t,$py_sfem_real_t,$py_sfem_real_t,$py_sfem_real_t"
