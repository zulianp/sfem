#!/usr/bin/env bash
# run with ./viz_decomposition.sh

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

SFEM_DIR=$SCRIPTPATH/../..
export PATH=$SFEM_DIR:$PATH
export PATH=$SFEM_DIR/python/sfem:$PATH
export PATH=$SFEM_DIR/python/sfem/mesh:$PATH

if (($# != 1))
then
	printf "usage: $0 <folder>\n" 1>&2
	exit -1
fi

folder=$1
files=(`ls -d $folder/part_*`)


set -x
for f in ${files[@]}
do
	echo $f
	raw_to_db.py $f "$f".vtk --point_data="$f/frank.raw,$f/neigh_count.raw" --point_data_type="float32,float32"
done
