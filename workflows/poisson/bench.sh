#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH

HERE=$PWD

export SFEM_ENABLE_OUTPUT=0
export OMP_NUM_THREADS=8
export SFEM_BASE_RESOLUTION=80
export SFEM_VERBOSE=0
export SFEM_ELEM_TYPE=TET4 
export SFEM_PROMOTE_TO_P2=1

elements_per_pack=(512 1024 2048 4096)

rm -f bench_*.csv

for e in ${elements_per_pack[@]}
do
	echo "Running with SFEM_ELEMENTS_PER_PACK=$e"
	SFEM_ELEMENTS_PER_PACK=$e SFEM_OPERATOR="PackedLaplacian" SMESH_TRACE_FILE=bench_packed_$e.csv poisson
done

SFEM_OPERATOR="PackedLaplacian" SMESH_TRACE_FILE=bench_packed_max.csv poisson
SFEM_OPERATOR="Laplacian"       SMESH_TRACE_FILE=bench_standard.csv   poisson

grep "Laplacian::apply" bench_*.csv | tr ',' ' ' | awk '{print $1,$3}'
