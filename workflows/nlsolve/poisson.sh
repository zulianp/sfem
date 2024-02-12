#!/usr/bin/env bash

set -e
set -x

export PATH=$CODE_DIR/sfem/python/sfem/mesh:$PATH
export PATH=$CODE_DIR/utopia/utopia/build:$PATH

export SFEM_MESH_DIR=$CODE_DIR/sfem/data/benchmarks/1_darcy_cube/mesh 
export SFEM_DIRICHLET_NODES=$CODE_DIR/sfem/data/benchmarks/1_darcy_cube/mesh/sidesets_aos/sfront.raw  
export SFEM_NEUMAN_FACES=$CODE_DIR/sfem/data/benchmarks/1_darcy_cube/mesh/sidesets_aos/sback.raw  
export SFEM_OUTPUT_DIR=$PWD/poisson_out

utopia_exec -app nlsolve -path $CODE_DIR/sfem/isolver_sfem.dylib  -solver_type ConjugateGradient --verbose --apply_gradient_descent_step

raw_to_db.py $SFEM_MESH_DIR $SFEM_OUTPUT_DIR/x.vtk -p "$SFEM_OUTPUT_DIR/out.raw"
