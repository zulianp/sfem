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

if (($# != 2))
then
	printf "usage: $0 <mesh.e> <output_folder>\n" 1>&2
	exit -1
fi

# set -x

# Volume mesh
mesh_path=$1
output_folder=$2
idx_type_size=4

exodusII_to_raw.py $mesh_path $output_folder

sidesets=(`ls $output_folder/sidesets`)
aos_folder=$output_folder/sidesets_aos

mkdir -p $aos_folder

for ss in ${sidesets[@]}
do
	name=`basename $ss`
	echo "> $name"
	soa_to_aos "$output_folder/sidesets/$ss/$name.*.raw" $idx_type_size $aos_folder/"$name".raw
done
