#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/sfem/mesh:$PATH

if (($# != 8))
then
	printf "usage: $0 <cell_type> <nx> <ny> <nz> <width> <height> <depth> <output_dir>\n" 1>&2
	exit -1
fi

cell_type=$1

x=$2
y=$3
z=$4

width=$5
height=$6
depth=$7

output_dir=$8

tempdir=`mktemp -d`

box_mesh.py $tempdir 		\
	--cell_type=$cell_type 		\
	-x $x -y $y -z $z  			\
	--width=$width --height=$height --depth=$depth

sfc $tempdir $output_dir
rm -rf $tempdir

raw_to_db.py $output_dir $output_dir/mesh.vtk   

