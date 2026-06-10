#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH

HERE=$PWD
export SFEM_OPERATOR="PackedLaplacian"
# export SFEM_OPERATOR="Laplacian"

export SFEM_BASE_RESOLUTION=80
export SFEM_ELEMENTS_PER_PACK=1024

rm -rf output_poisson

$LAUNCH poisson
raw_to_db output_poisson/mesh output_poisson.vtk  -p 'output_poisson/*.*' $EXTRA_OPTIONS
