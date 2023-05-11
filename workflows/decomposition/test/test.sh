#!/usr/bin/env bash

source ../../sfem_config.sh

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../../:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../data/benchmarks/meshes:$PATH

export DYLD_LIBRARY_PATH=$METIS_DIR/lib

num_procs=8

mesh_db=sphere.vtk
mesh=mesh
matrix=system
output=decomp.raw
test_output=decomp.txt

sphere.py $mesh_db 1
db_to_raw.py $mesh_db $mesh

mkdir -p $matrix

SFEM_HANDLE_NEUMANN=0 \
SFEM_HANDLE_RHS=0 \
SFEM_HANDLE_DIRICHLET=0 \
assemble $mesh $matrix

set -x
partition_mesh_based_on_operator $mesh $matrix $num_procs $output


python3 -c "import numpy as np; a = np.fromfile(\"$output\", dtype=np.int32).astype(np.float32).tofile(\"decomp-float32.raw\")"
raw_to_db.py $mesh decomp-vis.vtk --cell_data="decomp-float32.raw" --cell_data_type="float32"

raw2text.py $output int32 $test_output
