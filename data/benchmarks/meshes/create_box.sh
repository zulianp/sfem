#!/usr/bin/env bash

set -e

if [[ -z $SFEM_DIR ]]
then
	echo "SFEM_DIR must be defined with the installation prefix of sfem"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH
export PATH=$SFEM_DIR/bin:$PATH
export PATH=$SFEM_DIR/scripts/sfem/mesh:$PATH
export PYTHONPATH=$SFEM_DIR/lib:$SFEM_DIR/scripts:$PYTHONPATH
source $SFEM_DIR/workflows/sfem_config.sh


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

