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

create_box_2D.sh 4 40 20
rm mesh/z.raw

export SFEM_MESH_DIR=mesh

sleft=$SFEM_MESH_DIR/sidesets_aos/sleft.raw
sright=$SFEM_MESH_DIR/sidesets_aos/sright.raw

export PATH=$CODE_DIR/utopia/utopia/build_debug:$PATH
# export PATH=$CODE_DIR/utopia/utopia/build:$PATH

set -x

export VAR_UC=0
export VAR_UX=1
export VAR_UY=2
export BLOCK_SIZE=3

export SFEM_DIRICHLET_NODESET="$sleft,$sleft,$sleft,$sright"
# export SFEM_DIRICHLET_VALUE="0,0,3"
# export SFEM_DIRICHLET_VALUE="0,0,0"
export SFEM_DIRICHLET_VALUE="0,0,0,0.000001"
export SFEM_DIRICHLET_COMPONENT="$VAR_UC,$VAR_UX,$VAR_UY,$VAR_UX"

export SFEM_SHEAR_MODULUS="2.23"
export SFEM_FIRST_LAME_PARAMETER="3.35"
export SFEM_FRACTURE_TOUGHNESS="0.27"
export SFEM_LENGTH_SCALE="1"


export SFEM_OUTPUT_DIR=sfem_output


export SFEM_DEBUG_DUMP=1
# lldb -- 
utopia_exec -app nlsolve -path $CODE_DIR/sfem/franetg_plugin.dylib -solver_type Newton --verbose -max_it 40 --linear_start -damping 0.8

aos_to_soa $SFEM_OUTPUT_DIR/out.raw 8 $BLOCK_SIZE $SFEM_OUTPUT_DIR/out
raw_to_db.py $SFEM_MESH_DIR $SFEM_OUTPUT_DIR/x.vtk -p "$SFEM_OUTPUT_DIR/out.*.raw"


# MATRIXIO_DENSE_OUTPUT=1 print_crs $SFEM_OUTPUT_DIR/H_debug_0.raw/rowptr.raw $SFEM_OUTPUT_DIR/H_debug_0.raw/colidx.raw $SFEM_OUTPUT_DIR/H_debug_0.raw/values.raw int int double
# print_array $SFEM_OUTPUT_DIR/g_debug_0.raw double