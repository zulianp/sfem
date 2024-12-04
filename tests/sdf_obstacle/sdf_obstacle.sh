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

$LAUNCH $HERE/sdf_obstacle.py input.yaml

cd -

raw_to_db.py $CASE_DIR/mesh out.vtk -p "$CASE_DIR/output/*.raw"
