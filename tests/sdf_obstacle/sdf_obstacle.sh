#!/usr/bin/env bash

set -e

if (($# != 1))
then
	echo "usage $0 <case_dir>"
	exit 1
fi

HERE=$PWD
CASE_DIR=$1
cd $CASE_DIR

export SFEM_HEX8_ASSUME_AFFINE=1
# export SFEM_FIRST_LAME_PARAMETER=3.333
# export SFEM_SHEAR_MODULUS=0.357

$LAUNCH $HERE/sdf_obstacle.py input.yaml

cd -

raw_to_db.py $CASE_DIR/mesh out.vtk -p "$CASE_DIR/output/*.raw"
