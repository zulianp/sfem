#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH

# Remove me!
UTOPIA_EXEC=$CODE_DIR/utopia/utopia/build/utopia_exec

if [[ -z "$UTOPIA_EXEC" ]]
then
	echo "Error! Please define UTOPIA_EXEC=<path_to_utopia_exectuable>"
	exit -1
fi

solve()
{
	mat=$1
	rhs=$2
	x=$3
	# mpirun $UTOPIA_EXEC -app ls_solve -A $mat -b $rhs -out $x -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 -out $sol --verbose
	mpiexec -np 8 $UTOPIA_EXEC -app ls_solve -A $mat -b $rhs -out $x -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 --verbose
}

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
# Set velocity to zero on surface nodes
################################################

mux=$workspace/mux.raw
muy=$workspace/muy.raw
muz=$workspace/muz.raw

smask $nodes_to_zero $ux $mux 0
smask $nodes_to_zero $uy $muy 0
smask $nodes_to_zero $uz $muz 0

################################################
# Compute (div(u), test)_L2
################################################

divu=$workspace/divu.raw
divergence $mesh_path $mux $muy $muz $divu

################################################
# Solve linear system
################################################

pressure=$workspace/pressure.raw
solve $workspace/rowptr.raw $divu $pressure

################################################
# Compute gradients
################################################

# per Cell quantities
dpdx=$workspace/dpdx.raw
dpdy=$workspace/dpdy.raw
dpdz=$workspace/dpdz.raw

cgrad $mesh_path $pressure $dpdx $dpdy $dpdz

################################################
# P0 to P1 projection
################################################

# TODO

################################################
# Compute WSS
################################################

# TODO

################################################
# Visualize WSS
################################################

# TODO

# convert final output
raw2mesh.py -d $mesh_path --field=$pressure --field_dtype=float64 --output=$workspace/pressure.vtu
