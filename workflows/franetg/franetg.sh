#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH


create_box_2D.sh 8 40 20

# export PATH=$CODE_DIR/utopia/utopia/build:$PATH


# export SFEM_MESH_DIR=$CODE_DIR/sfem/data/benchmarks/4_rectangle/mesh
# export SFEM_DIRICHLET_NODES=$SFEM_MESH_DIR/sidesets_aos/sfront.raw  
# export SFEM_NEUMAN_FACES=$CODE_DIR/sfem/data/benchmarks/1_darcy_cube/mesh/sidesets_aos/sback.raw  
# export SFEM_OUTPUT_DIR=$PWD/poisson_out

# utopia_exec -app nlsolve -path $CODE_DIR/sfem/franetg_plugin.dylib -solver_type Newton --verbose

# raw_to_db.py $SFEM_MESH_DIR $SFEM_OUTPUT_DIR/x.vtk -p "$SFEM_OUTPUT_DIR/out.raw"
