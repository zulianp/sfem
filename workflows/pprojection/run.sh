#!/usr/bin/env bash

# Methods 
# - Average pressure 0
# - Fix 1 point
# - ?

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

real_type_size=8

velpath=/Users/patrickzulian/Desktop/code/sfem/data/run-hsp-4
gridux=$velpath/0.24450.raw
griduy=$velpath/1.24450.raw
griduz=$velpath/2.24450.raw

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
python3 -c "import numpy as np; np.array([7699704]).astype(np.int32).tofile('dirichlet.raw')"
# fix_value=12500.4
fix_value=0

dirichlet_nodes=$workspace/dirichlet.raw
mv dirichlet.raw $dirichlet_nodes

SFEM_HANDLE_DIRICHLET=1 \
SFEM_DIRICHLET_NODES=$dirichlet_nodes \
assemble $mesh_path $workspace

# NEXT SHOULD BE A LOOP OVER FILES

################################################
# Transfer from grid to mesh
################################################

ux=$workspace/ux.raw
uy=$workspace/uy.raw
uz=$workspace/uz.raw

SFEM_READ_FP32=1 pizzastack_to_mesh $nx $ny $nz $gridux $mesh_path $ux
SFEM_READ_FP32=1 pizzastack_to_mesh $nx $ny $nz $griduy $mesh_path $uy
SFEM_READ_FP32=1 pizzastack_to_mesh $nx $ny $nz $griduz $mesh_path $uz

################################################
# Set velocity to zero on surface nodes
################################################

mux=$workspace/mux.raw
muy=$workspace/muy.raw
muz=$workspace/muz.raw

smask $nodes_to_zero $ux $mux 0
smask $nodes_to_zero $uy $muy 0
smask $nodes_to_zero $uz $muz 0

# CHECK BEGIN
# check=$workspace/check_mask.raw
# smask $nodes_to_zero $ux $check 666
# raw2mesh.py -d $mesh_path --field=$check --field_dtype=float64 --output=$workspace/check_mask.vtk
# CHECK END

################################################
# Compute (div(u), test)_L2
################################################

divu=$workspace/divu.raw
SFEM_SCALE=-1 divergence $mesh_path $mux $muy $muz $divu

# usage: ./lumped_mass_inv <folder> <in.raw> <out.raw>
cdivu=$workspace/cdivu.raw
lumped_mass_inv $mesh_path $divu $cdivu

# CHECK BEGIN
raw2mesh.py -d $mesh_path --field=$cdivu --field_dtype=float64 --output=$workspace/cdivu.vtk
# CHECK END

# dirichlet_values=$workspace/dirichlet_values.raw
# sgather $dirichlet_nodes $real_type_size $cdivu $dirichlet_values

################################################
# Solve linear system
################################################

# Fix degree of freedom for uniqueness of solution
rhs=$workspace/rhs.raw
smask $workspace/dirichlet.raw $divu $rhs $fix_value

# rhs=$workspace/rhs.raw
# soverride $dirichlet_nodes $real_type_size $dirichlet_values ? $rhs

potential=$workspace/potential.raw
solve $workspace/rowptr.raw $rhs $potential

# CHECK BEGIN
raw2mesh.py -d $mesh_path --field=$potential --field_dtype=float64 --output=$workspace/potential.vtk
# CHECK END

################################################
# Compute gradients
################################################

# Per Cell quantities
p0_dpdx=$workspace/p0_dpdx.raw
p0_dpdy=$workspace/p0_dpdy.raw
p0_dpdz=$workspace/p0_dpdz.raw

# coefficients: P1 -> P0
cgrad $mesh_path $potential $p0_dpdx $p0_dpdy $p0_dpdz

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

#
raw2mesh.py -d $mesh_path --field=$p1_dpdx --field_dtype=float64 --output=$workspace/velx.vtk
raw2mesh.py -d $mesh_path --field=$p1_dpdy --field_dtype=float64 --output=$workspace/vely.vtk
raw2mesh.py -d $mesh_path --field=$p1_dpdz --field_dtype=float64 --output=$workspace/velz.vtk

################################################
# Compute WSS
################################################

vshear_prefix=$workspace/volshear
sshear_prefix=$workspace/surfshear

# coefficients: P1 -> P0
cshear $mesh_path $p1_dpdx $p1_dpdy $p1_dpdz $vshear_prefix

# Map shear to surface elements

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
raw2surfmesh.py -d $surf_mesh_path --cell_data=$wssmag --cell_data_dtype=float64 --output=$workspace/wssmag.vtk

################################################
# Project stress to P1
################################################

# wssmag_p1=$workspace/wssmag_p1.raw
# surf_projection_p0_to_p1 $surf_mesh_path $wssmag $wssmag_p1

# # Convert final output (P1)
# raw2surfmesh.py -d $surf_mesh_path --field=$wssmag_p1 --field_dtype=float64 --output=$workspace/wssmag_p1.vtk

