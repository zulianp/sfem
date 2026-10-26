#!/usr/bin/env bash

set -e

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

mkdir -p test_actual
mkdir -p test_oracle

export SFEM_HANDLE_DIRICHLET=0
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=true 

# CUDA + OpenMP
$LAUNCH assemble $CASE test_actual;  

# Serial
$LAUNCH assemble_oracle $CASE test_oracle 

fdiff.py test_actual/values.raw test_oracle/values.raw


# MATRIXIO_DENSE_OUTPUT=1 ../../matrix.io/print_crs test_actual/rowptr.raw test_actual/colidx.raw test_actual/values.raw int int double
# MATRIXIO_DENSE_OUTPUT=1 ../../matrix.io/print_crs test_oracle/rowptr.raw test_oracle/colidx.raw test_oracle/values.raw int int double
