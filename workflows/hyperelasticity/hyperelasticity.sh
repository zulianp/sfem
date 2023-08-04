#!/usr/bin/env bash

# set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH
PATH=$SCRIPTPATH/../../../matrix.io:$PATH

# create_cylinder.sh 1

export SFEM_MESH_DIR=mesh

sleft=$SFEM_MESH_DIR/sidesets_aos/sinlet.raw
sright=$SFEM_MESH_DIR/sidesets_aos/soutlet.raw

export PATH=$CODE_DIR/utopia/utopia/build_debug:$PATH

set -x

export VAR_UX=0
export VAR_UY=1
export VAR_UZ=2
export BLOCK_SIZE=3

export SFEM_DIRICHLET_NODESET="$sleft,$sleft,$sleft,$sright"
export SFEM_DIRICHLET_VALUE="0,0,0,0.00001"
export SFEM_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX"

export SFEM_SHEAR_MODULUS="1"
export SFEM_FIRST_LAME_PARAMETER="1"

export SFEM_OUTPUT_DIR=sfem_output

utopia_exec -app nlsolve -path $CODE_DIR/sfem/hyperelasticity_plugin.dylib -solver_type Newton --verbose -max_it 1	

aos_to_soa $SFEM_OUTPUT_DIR/out.raw 8 $BLOCK_SIZE $SFEM_OUTPUT_DIR/out
raw_to_db.py $SFEM_MESH_DIR $SFEM_OUTPUT_DIR/x.vtk -p "$SFEM_OUTPUT_DIR/out.*.raw"
