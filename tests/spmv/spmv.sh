#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../../workflows/sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../..:$PATH
export PATH=$SCRIPTPATH/../../python/sfem:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/utils:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

LAUNCH="mpiexec -np 8"
# LAUNCH=""

source venv/bin/activate

mesh=mesh
mesh_sorted=mesh_sorted
create_sphere.sh 1
sfc $mesh $mesh_sorted


touch $mesh_sorted/zd.raw
touch $mesh_sorted/on.raw


mkdir -p linear_system
assemble $mesh_sorted linear_system


eval_nodal_function.py "x" $mesh_sorted/x.raw $mesh_sorted/y.raw  $mesh_sorted/z.raw linear_system/rhs.raw

 # ../../spmv <alpha> <transpose> <crs_folder> <x.raw> <output.raw> <output_folder>
spmv 1 0 linear_system linear_system/rhs.raw test.raw


deactivate