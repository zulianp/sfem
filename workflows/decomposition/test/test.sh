#!/usr/bin/env bash

set -e

if [[ -z $SFEM_DIR ]]
then
	echo "SFEM_DIR must be defined with the installation prefix of sfem"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH
export PATH=$SFEM_DIR/bin:$PATH
export PATH=$SFEM_DIR/scripts/sfem/mesh:$PATH
export PYTHONPATH=$SFEM_DIR/lib:$SFEM_DIR/scripts:$PYTHONPATH
source $SFEM_DIR/workflows/sfem_config.sh
export PATH=$SCRIPTPATH/../../../data/benchmarks/meshes:$PATH

export DYLD_LIBRARY_PATH=$METIS_DIR/lib

num_procs=8

mesh_db=sphere.vtk
mesh=mesh
matrix=system
output=decomp.raw
test_output=decomp.txt

# sphere.py $mesh_db --refinements=5
# db_to_raw.py $mesh_db $mesh --select_elem_type=tetra
# refine $mesh refined


export OMP_NUM_THREAD=2
export OMP_PROC_BIND=true
time mpiexec -np 4 partition refined partitioned

# mkdir -p $matrix

# SFEM_HANDLE_NEUMANN=0 \
# SFEM_HANDLE_RHS=0 \
# SFEM_HANDLE_DIRICHLET=0 \
# assemble $mesh $matrix

# partition_mesh_based_on_operator $mesh $matrix $num_procs $output
# fp_convert.py $output proc-float32.raw int32 float32
# raw_to_db.py $mesh decomp-vis.vtk --cell_data="proc-float32.raw" --cell_data_type="float32"

# raw2text.py $output int32 $test_output
