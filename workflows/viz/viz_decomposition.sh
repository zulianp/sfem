#!/usr/bin/env bash
# run with ./viz_decomposition.sh

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

SFEM_DIR=$SCRIPTPATH/../..
export PATH=$SFEM_DIR:$PATH
export PATH=$SFEM_DIR/python:$PATH
export PATH=$SFEM_DIR/python/mesh:$PATH

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
	raw_to_db.py $f "$f".vtk --point_data="$f"/frank.raw --point_data_type=float32
done
