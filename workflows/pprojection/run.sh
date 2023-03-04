#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH

set -x

mkdir -p workspace
workspace="workspace"

################################################
# Grid
################################################
nx=480
ny=672
nz=736

gridux=/Users/patrickzulian/Desktop/code/sfem/data/veldata.raw 
griduy=$gridux
griduz=$gridux

################################################
# Mesh
################################################
mesh_path=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better 

# >

################################################
# Boundary dofs
################################################
surf_mesh_path=$workspace/skinned

mkdir -p $surf_mesh_path

skin $mesh_path $surf_mesh_path
nodes_to_zero=$surf_mesh_path/node_mapping.raw

################################################
# Assemble laplacian
################################################

# List dirchlet nodes
python3 -c "import numpy as np; np.array([0]).astype(np.int32).tofile('dn.raw')"

mv dn.raw $workspace/dn.raw

SFEM_HANDLE_DIRICHLET=1 \
SFEM_DIRICHLET_NODES=$workspace/dn.raw \
assemble $mesh_path $workspace

# NEXT SHOULD BE A LOOP OVER FILES

################################################
# Transfer from grid to mesh
################################################

ux=$workspace/ux.raw
uy=$workspace/uy.raw
uz=$workspace/uz.raw

# SFEM_READ_FP32=1 pizzastack_to_mesh $nx $ny $nz $gridux $mesh_path $ux
# SFEM_READ_FP32=1 pizzastack_to_mesh $nx $ny $nz $griduy $mesh_path $uy
# SFEM_READ_FP32=1 pizzastack_to_mesh $nx $ny $nz $griduz $mesh_path $uz

################################################
# Compute (div(u), test)_L2
################################################

divu=$workspace/divu.raw
divergence $mesh_path $ux $uy $uz $divu

# TODO
# convert final output
raw2mesh.py -d $mesh_path --field=$divu --field_dtype=float64 --output=$workspace/out.vtu
