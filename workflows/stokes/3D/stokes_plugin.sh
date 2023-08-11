#!/usr/bin/env bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

set -e

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../data/benchmarks/meshes:$PATH
PATH=$SCRIPTPATH/../../../../matrix.io:$PATH
PATH=$SCRIPTPATH/../../convert_exodus_to_raw:$PATH

# export PATH=$CODE_DIR/utopia/utopia/build_debug:$PATH
export PATH=$CODE_DIR/utopia/utopia/build:$PATH

export VAR_UX=0
export VAR_UY=1
export VAR_UZ=2
export VAR_P=3

export BLOCK_SIZE=4

export SFEM_VISCOSITY="100"
export SFEM_MASS_DENSITY="1"
export SFEM_OUTPUT_DIR=sfem_output

##########################################################################################

db=/Users/patrickzulian/Desktop/cloud/owncloud_HSLU/Patrick/2023/Cases/FP1100/fluid.e
export SFEM_MESH_DIR=mesh
# convert_exodus_to_raw.sh $db $SFEM_MESH_DIR

sinlet=$SFEM_MESH_DIR/sidesets_aos/LEFT_OPENING.raw
soutlet=$SFEM_MESH_DIR/sidesets_aos/RIGHT_OPENING.raw
swall=$SFEM_MESH_DIR/sidesets_aos/WALLS.raw
smembrane=$SFEM_MESH_DIR/sidesets_aos/ARTIFICIAL_MEMBRANE.raw
pdof=pdof.raw

head -c 4 $SFEM_MESH_DIR/sidesets_aos/ARTIFICIAL_MEMBRANE.raw > $pdof

# export SFEM_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$soutlet,$soutlet,$soutlet,$swall,$swall,$swall,$smembrane"
# export SFEM_DIRICHLET_VALUE="0.1,0,0,0.1,0,0,0,0,0,0"
# export SFEM_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_P"


# export SFEM_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$soutlet,$soutlet,$soutlet,$swall,$swall,$swall,$smembrane,$smembrane,$smembrane,$pdof"
# export SFEM_DIRICHLET_VALUE="0.1,0,0,0.1,0,0,0,0,0,0,0,0,0"
# export SFEM_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_P"


export SFEM_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$swall,$swall,$swall,$smembrane,$smembrane,$smembrane,$soutlet"
export SFEM_DIRICHLET_VALUE="0.1,0,0,0,0,0,0,0,0,0"
export SFEM_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_P"

##########################################################################################

# create_cylinder.sh 3
# sinlet=$SFEM_MESH_DIR/sidesets_aos/sinlet.raw
# soutlet=$SFEM_MESH_DIR/sidesets_aos/soutlet.raw
# swall=$SFEM_MESH_DIR/sidesets_aos/swall.raw
# export SFEM_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$soutlet,$soutlet,$soutlet,$swall,$swall,$swall"
# export SFEM_DIRICHLET_VALUE="1,0,0,1,0,0,0,0,0"
# export SFEM_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ"

##########################################################################################

set -x

# lldb --  
utopia_exec -app nlsolve -path $CODE_DIR/sfem/stokes_plugin.dylib -solver_type ConjugateGradient --verbose -max_it 10000 -matrix_free true -apply_gradient_descent_step true
# lldb -- 
# utopia_exec --verbose  -app nlsolve -path $CODE_DIR/sfem/stokes_plugin.dylib -solver_type Newton -max_it 40 -ksp_monitor -pc_monitor

aos_to_soa $SFEM_OUTPUT_DIR/out.raw 8 $BLOCK_SIZE $SFEM_OUTPUT_DIR/x
raw_to_db.py $SFEM_MESH_DIR $SFEM_OUTPUT_DIR/x.vtk -p "$SFEM_OUTPUT_DIR/x.*.raw"
