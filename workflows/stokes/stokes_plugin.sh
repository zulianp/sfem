#!/usr/bin/env bash

# set -e

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

create_box_2D.sh 6 2 1
rm mesh/z.raw


export SFEM_MESH_DIR=mesh

sleft=$SFEM_MESH_DIR/sidesets_aos/sleft.raw
sright=$SFEM_MESH_DIR/sidesets_aos/sright.raw
sbottom=$SFEM_MESH_DIR/sidesets_aos/sbottom.raw
stop=$SFEM_MESH_DIR/sidesets_aos/stop.raw

# export PATH=$CODE_DIR/utopia/utopia/build_debug:$PATH
export PATH=$CODE_DIR/utopia/utopia/build:$PATH

set -x

export VAR_UX=0
export VAR_UY=1
export VAR_P=2

export BLOCK_SIZE=3

export SFEM_DIRICHLET_NODESET="$sleft,$sleft,$sright,$sright,$sbottom,$sbottom,$stop,$stop"
export SFEM_DIRICHLET_VALUE="0.1,0,0.1,0,0,0,0,0"
export SFEM_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UX,$VAR_UY,$VAR_UX,$VAR_UY,$VAR_UX,$VAR_UY"

export SFEM_VISCOSITY="1"
export SFEM_MASS_DENSITY="1"

export SFEM_OUTPUT_DIR=sfem_output
export SFEM_MATERIAL=linear

# lldb --  
# utopia_exec -app nlsolve -path $CODE_DIR/sfem/stokes_plugin.dylib -solver_type ConjugateGradient --verbose -max_it 10000 -matrix_free false -apply_gradient_descent_step true
utopia_exec -app nlsolve -path $CODE_DIR/sfem/stokes_plugin.dylib -solver_type Newton --verbose --max_it 1 -ksp_type preonly -pc_type lu

aos_to_soa $SFEM_OUTPUT_DIR/out.raw 8 $BLOCK_SIZE $SFEM_OUTPUT_DIR/out
raw_to_db.py $SFEM_MESH_DIR $SFEM_OUTPUT_DIR/x.vtk -p "$SFEM_OUTPUT_DIR/out.*.raw"
