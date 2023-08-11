#!/usr/bin/env bash

# set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../data/benchmarks/meshes:$PATH
PATH=$SCRIPTPATH/../../../../matrix.io:$PATH


# db=/Users/patrickzulian/Desktop/cloud/owncloud_HSLU/Patrick/2023/Cases/FP1100/fluid.e
export SFEM_MESH_DIR=mesh

# db_to_raw.py $db $SFEM_MESH_DIR
# 
# create_cylinder.sh 1
# SFEM_MESH_DIR=/Users/patrickzulian/Desktop/code/sfem/data/benchmarks/1_darcy_cube/mesh
nvars=4


sinlet=$SFEM_MESH_DIR/sidesets_aos/sinlet.raw
soutlet=$SFEM_MESH_DIR/sidesets_aos/soutlet.raw
swall=$SFEM_MESH_DIR/sidesets_aos/swall.raw

# export PATH=$CODE_DIR/utopia/utopia/build_debug:$PATH
export PATH=$CODE_DIR/utopia/utopia/build:$PATH

set -x

export VAR_UX=0
export VAR_UY=1
export VAR_UZ=2
export VAR_P=3

export BLOCK_SIZE=4

export SFEM_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$soutlet,$soutlet,$soutlet,$swall,$swall,$swall"
export SFEM_DIRICHLET_VALUE="1,0,0,1,0,0,0,0,0"
export SFEM_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ"

export SFEM_VISCOSITY="1"
export SFEM_MASS_DENSITY="1"

export SFEM_OUTPUT_DIR=sfem_output
export SFEM_MATERIAL=linear

# lldb --  
# utopia_exec -app nlsolve -path $CODE_DIR/sfem/stokes_plugin.dylib -solver_type ConjugateGradient --verbose -max_it 10000 -matrix_free false -apply_gradient_descent_step true
utopia_exec -app nlsolve -path $CODE_DIR/sfem/stokes_plugin.dylib -solver_type Newton --verbose --max_it 1 -ksp_type preonly -pc_type lu

aos_to_soa $SFEM_OUTPUT_DIR/out.raw 8 $BLOCK_SIZE $SFEM_OUTPUT_DIR/x
raw_to_db.py $SFEM_MESH_DIR $SFEM_OUTPUT_DIR/x.vtk -p "$SFEM_OUTPUT_DIR/x.*.raw"
