#!/usr/bin/env bash

set -e
set -x

CASE=../../fetest/meshes/case4
CASE=../../fetest/meshes/case5
CASE=../data/mesh-multi-outlet-better/
# CASE=../sorted_mesh
# CASE=../data/half_tera_cube 

export SFEM_HANDLE_DIRICHLET=0
export OMP_NUM_THREADS=4 
export OMP_PROC_BIND=true 

# CUDA + OpenMP
../assemble $CASE test_actual;  

# Serial
../assemble_oracle $CASE test_oracle 

../python/fdiff.py test_actual/values.raw test_oracle/values.raw


# MATRIXIO_DENSE_OUTPUT=1 ../../matrix.io/print_crs test_actual/rowptr.raw test_actual/colidx.raw test_actual/values.raw int int double
# MATRIXIO_DENSE_OUTPUT=1 ../../matrix.io/print_crs test_oracle/rowptr.raw test_oracle/colidx.raw test_oracle/values.raw int int double