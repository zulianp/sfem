#!/usr/bin/env bash

set -e

# source $CODE_DIR/merge_git_repos/sfem/venv/bin/activate
# export SFEM_PATH=$INSTALL_DIR/sfem

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH
export PATH=$SFEM_PATH/scripts/sfem/mesh/:$PATH
export PATH=$SFEM_PATH/scripts/sfem/grid/:$PATH
export PATH=$SFEM_PATH/scripts/sfem/sdf/:$PATH
export PATH=$SFEM_PATH/worflows/mech/:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

HERE=$PWD

export SFEM_ENABLE_OUTPUT=0
# export SFEM_BASE_RESOLUTION=300
export SFEM_BASE_RESOLUTION=200

SFEM_OPERATOR="PackedLaplacian" SFEM_TRACE_FILE=bench_packed.csv   poisson

SFEM_OPERATOR="Laplacian"       SFEM_TRACE_FILE=bench_standard.csv poisson

grep "Laplacian::apply" bench_*.csv | tr ',' ' ' | awk '{print $3}'
