#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH

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

p1grads()
{
	if (($# != 4))
	then
		printf "usage: $0 <potential.raw> <outx.raw> <outy.raw> <outz.raw>\n" 1>&2
		exit -1
	fi

	potential=$1
	p1_dpdx=$2
	p1_dpdy=$3
	p1_dpdz=$4

	# Per Cell quantities
	p0_dpdx=$workspace/temp_p0_dpdx.raw
	p0_dpdy=$workspace/temp_p0_dpdy.raw
	p0_dpdz=$workspace/temp_p0_dpdz.raw

	# coefficients: P1 -> P0
	cgrad $mesh_path $potential $p0_dpdx $p0_dpdy $p0_dpdz

	################################################
	# P0 to P1 projection
	################################################

	# coefficients: P0 -> P1
	projection_p0_to_p1 $mesh_path $p0_dpdx $p1_dpdx
	projection_p0_to_p1 $mesh_path $p0_dpdy $p1_dpdy
	projection_p0_to_p1 $mesh_path $p0_dpdz $p1_dpdz
}

set -x

mkdir -p workspace
workspace="workspace"
real_type_size=8

################################################
# Mesh
################################################

mesh_path=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better

################################################
# Boundary nodes
################################################

# surf_mesh_path=$workspace/skinned

# mkdir -p $surf_mesh_path

# skin $mesh_path $surf_mesh_path
# surface_nodes=$surf_mesh_path/node_mapping.raw
# parent_elements=$surf_mesh_path/parent.raw

################################################
# Split wall from inlet and outlet
################################################

# nodes_to_zero=$workspace/wall_idx.raw

# set_diff $surface_nodes $mesh_path/zd.raw $workspace/temp.raw
# set_diff $workspace/temp.raw $mesh_path/on.raw $nodes_to_zero

################################################
# Assemble laplacian
################################################

dirichlet_nodes=$mesh_path/zd.raw
# neumann_faces=$mesh_path/on.raw # FIXME

SFEM_HANDLE_DIRICHLET=1 \
SFEM_DIRICHLET_NODES=$dirichlet_nodes \
SFEM_HANDLE_NEUMANN=1 \
assemble $mesh_path $workspace

potential=$workspace/potential.raw
solve $workspace/rowptr.raw $workspace/rhs.raw $potential

################################################
# Compute divergence
################################################

vel_x=$workspace/vel_x.raw
vel_y=$workspace/vel_y.raw
vel_z=$workspace/vel_z.raw
divu=$workspace/divu.raw

p1grads $potential $vel_x $vel_y $vel_z
divergence $mesh_path $vel_x $vel_y $vel_z $divu
cdivu=$workspace/cdivu.raw
lumped_mass_inv $mesh_path $divu $cdivu

# 
raw2mesh.py -d $mesh_path --field=$cdivu --field_dtype=float64 --output=$workspace/cdivu.vtk
raw_to_db.py $mesh_path workspace/gradient.vtk --point_data="$workspace/vel_*.raw"

