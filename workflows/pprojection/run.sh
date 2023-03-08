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
# Boundary nodes
################################################
surf_mesh_path=$workspace/skinned

mkdir -p $surf_mesh_path

skin $mesh_path $surf_mesh_path
surface_nodes=$surf_mesh_path/node_mapping.raw
parent_elements=$surf_mesh_path/parent.raw

################################################
# Split wall from inlet and outlet
################################################

nodes_to_zero=$workspace/wall_idx.raw

set_diff $surface_nodes $mesh_path/zd.raw $workspace/temp.raw
set_diff $workspace/temp.raw $mesh_path/on.raw $nodes_to_zero

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

# Per Cell quantities
p0_dpdx=$workspace/p0_dpdx.raw
p0_dpdy=$workspace/p0_dpdy.raw
p0_dpdz=$workspace/p0_dpdz.raw

# coefficients: P1 -> P0
cgrad $mesh_path $pressure $p0_dpdx $p0_dpdy $p0_dpdz

################################################
# P0 to P1 projection
################################################

# Per Node quantities
p1_dpdx=$workspace/p1_dpdx.raw
p1_dpdy=$workspace/p1_dpdy.raw
p1_dpdz=$workspace/p1_dpdz.raw

# coefficients: P0 -> P1
projection_p0_to_p1 $mesh_path $p0_dpdx $p1_dpdx
projection_p0_to_p1 $mesh_path $p0_dpdy $p1_dpdy
projection_p0_to_p1 $mesh_path $p0_dpdz $p1_dpdz

################################################
# Compute WSS
################################################

vshear_prefix=$workspace/volshear
sshear_prefix=$workspace/surfshear

# coefficients: P1 -> P0
cshear $mesh_path $p1_dpdx $p1_dpdy $p1_dpdz $vshear_prefix

# Map shear to surface elements
real_type_size=8
sgather $parent_elements $real_type_size $vshear_prefix".0.raw" $sshear_prefix".0.raw"
sgather $parent_elements $real_type_size $vshear_prefix".1.raw" $sshear_prefix".1.raw"
sgather $parent_elements $real_type_size $vshear_prefix".2.raw" $sshear_prefix".2.raw"
sgather $parent_elements $real_type_size $vshear_prefix".3.raw" $sshear_prefix".3.raw"
sgather $parent_elements $real_type_size $vshear_prefix".4.raw" $sshear_prefix".4.raw"
sgather $parent_elements $real_type_size $vshear_prefix".5.raw" $sshear_prefix".5.raw"

wssmag=$workspace/wssmag.raw

# coefficients: P0 -> P0
wss $surf_mesh_path $sshear_prefix $wssmag

################################################
# Visualize WSS
################################################

# Convert final output (P0)
raw2mesh.py -d $mesh_path --cell_data=$wssmag --cell_data_dtype=float64 --output=$workspace/wssmag.vtk
