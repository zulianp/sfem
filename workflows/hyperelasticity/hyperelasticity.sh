#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python/sfem:$PATH
PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH
PATH=$SCRIPTPATH/../../../matrix.io:$PATH

# create_cylinder.sh 6
# create_cylinder.sh 3

create_cylinder_p2.sh 4
export SFEM_USE_MACRO=1

export SFEM_MESH_DIR=mesh

sleft=$SFEM_MESH_DIR/sidesets_aos/sinlet.raw
sright=$SFEM_MESH_DIR/sidesets_aos/soutlet.raw

# export PATH=$CODE_DIR/utopia/utopia/build_debug:$PATH
export PATH=$CODE_DIR/utopia/utopia/build:$PATH

set -x

export VAR_UX=0
export VAR_UY=1
export VAR_UZ=2
export SFEM_BLOCK_SIZE=3

export SFEM_DIRICHLET_NODESET="$sleft,$sleft,$sleft,$sright"
export SFEM_DIRICHLET_VALUE="0,0,0,0.5"
export SFEM_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX"

# export SFEM_DIRICHLET_NODESET="$sleft,$sleft,$sleft,$sright,$sright,$sright"
# export SFEM_DIRICHLET_VALUE="0,0,0,0.05,0,0"
# export SFEM_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ"

export SFEM_SHEAR_MODULUS="1"
export SFEM_FIRST_LAME_PARAMETER="1"

export SFEM_OUTPUT_DIR=sfem_output
export SFEM_MATERIAL=linear

# export OMP_NUM_THREADS=32
export OMP_NUM_THREADS=16
# export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true


# lldb -- 
utopia_exec -app nlsolve -path $CODE_DIR/sfem/hyperelasticity_plugin.dylib -solver_type ConjugateGradient --verbose -max_it 10000 -apply_gradient_descent_step true -atol 1e-6 #-matrix_free false

aos_to_soa $SFEM_OUTPUT_DIR/out.raw 8 $SFEM_BLOCK_SIZE $SFEM_OUTPUT_DIR/out
raw_to_db.py $SFEM_MESH_DIR $SFEM_OUTPUT_DIR/x.vtk -p "$SFEM_OUTPUT_DIR/out.*.raw"
