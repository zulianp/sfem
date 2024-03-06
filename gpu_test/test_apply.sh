#!/usr/bin/env bash

set -e


SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../python:$PATH
PATH=$SCRIPTPATH/../python/mesh:$PATH


LAUNCH=srun
LAUNCH=""

if (($# != 1))
then
	printf "usage: $0 <mesh_path>\n" 1>&2
	exit -1
fi

set -x

CASE=$1
# lets just use this for now
# IN=`mktemp`
IN=debugme.raw
fp_convert.py $CASE/x.raw $IN float32 float64

mkdir -p test_actual
mkdir -p test_oracle

export SFEM_HANDLE_DIRICHLET=0
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 

# CUDA + OpenMP
$LAUNCH lapl $CASE $IN test_actual/lapl.raw;  

# Serial
$LAUNCH lapl_oracle $CASE $IN test_oracle/lapl.raw 

fdiff.py test_actual/lapl.raw test_oracle/lapl.raw


# MATRIXIO_DENSE_OUTPUT=1 ../../matrix.io/print_crs test_actual/rowptr.raw test_actual/colidx.raw test_actual/values.raw int int double
# MATRIXIO_DENSE_OUTPUT=1 ../../matrix.io/print_crs test_oracle/rowptr.raw test_oracle/colidx.raw test_oracle/values.raw int int double


rm $IN