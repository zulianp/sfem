#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../workflows/divergence:$PATH


UTOPIA_EXEC=$CODE_DIR/utopia/utopia/build/utopia_exec

if [[ -z "$UTOPIA_EXEC" ]]
then
	echo "Error! Please define UTOPIA_EXEC=<path_to_utopia_exectuable>"
	exit -1
fi

if (($# != 4))
then
	printf "usage: $0 <vx.raw> <vy.raw> <vz.raw> <output_dir>\n" 1>&2
	exit -1
fi

velx=$1
vely=$2
velz=$3
output=$4

ls -la $velx
ls -la $vely
ls -la $velz

solve()
{
	mat_=$1
	rhs_=$2
	x_=$3

	echo "rhs=$rhs_"
	mpiexec -np 8 $UTOPIA_EXEC -app ls_solve -A $mat_ -b $rhs_ -out $x_ -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 --verbose
}

norm_fp64()
{
	if (($# != 1))
	then
		printf "usage: $0 <vec.fp64.raw>\n" 1>&2
		exit -1
	fi

	vec=$1
	norm_vec=`python3 -c "import numpy as np; from numpy import linalg as la; a=np.fromfile(\"$vec\", dtype=np.float64); print(la.norm(a, 2))"`
	echo "norm: $norm_vec  (vec=$vec)"
}

p1grads()
{
	if (($# != 4))
	then
		printf "usage: $0 <potential.raw> <outx.raw> <outy.raw> <outz.raw>\n" 1>&2
		exit -1
	fi

	potential_=$1
	p1_dpdx_=$2
	p1_dpdy_=$3
	p1_dpdz_=$4

	# Per Cell quantities
	p0_dpdx_=$workspace/temp_p0_dpdx.raw
	p0_dpdy_=$workspace/temp_p0_dpdy.raw
	p0_dpdz_=$workspace/temp_p0_dpdz.raw

	# coefficients: P1 -> P0
	cgrad $mesh_path $potential_ $p0_dpdx_ $p0_dpdy_ $p0_dpdz_

	################################################
	# P0 to P1 projection
	################################################

	# coefficients: P0 -> P1
	projection_p0_to_p1 $mesh_path $p0_dpdx_ $p1_dpdx_
	projection_p0_to_p1 $mesh_path $p0_dpdy_ $p1_dpdy_
	projection_p0_to_p1 $mesh_path $p0_dpdz_ $p1_dpdz_
}

mesh_path=./mesh
workspace=`mktemp -d`

post_dir=$output
mkdir -p $post_dir

sides=$workspace/dirichlet.raw

python3 -c "import numpy as np; a=np.fromfile(\"$mesh_path/x.raw\", dtype=np.float32); a.fill(0); a.astype(np.float64).tofile(\"$sides\")"
volume_cylinder=`python3 -c "import numpy as np; print(f'{np.pi * 0.5 * 0.5 * 2}')"`
echo "measure(cylinder) = $volume_cylinder"

# Boundary nodes
boundary_wall=$mesh_path/sidesets_aos/swall.raw
boundary_inlet=$mesh_path/sidesets_aos/sinlet.raw
boundary_outlet=$mesh_path/sidesets_aos/soutlet.raw

# boundary_inout=$workspace/sinout.raw
# set_union $boundary_inlet $boundary_outlet $boundary_inout

# smask $boundary_wall $velx $velx 0
# smask $boundary_wall $vely $vely 0
# smask $boundary_wall $velz $velz 0

divu=$workspace/divu.raw
SFEM_VERBOSE=1 divergence $mesh_path $velx $vely $velz $divu

######################################
# Viz
######################################
cell_div=$workspace/cell_div.raw
cdiv $mesh_path $velx $vely $velz $cell_div

node_div=$workspace/node_div.raw
lumped_mass_inv $mesh_path $divu $node_div

cell_volume=$workspace/cell_volume.raw
volumes $mesh_path $cell_volume
######################################

dirichlet_nodes=$boundary_wall
# dirichlet_nodes=$boundary_inlet

rhs=$workspace/rhs_divu.raw
# smask $dirichlet_nodes $divu $rhs 0
smask $boundary_wall $divu $rhs 0
# smask $boundary_inlet $rhs $rhs 0
# smask $boundary_outlet $rhs $rhs 0

norm_fp64 $rhs

######################################
# Viz
######################################
rhs_viz=$workspace/crhs.raw
lumped_mass_inv $mesh_path $rhs $rhs_viz

smask $dirichlet_nodes $sides $sides 1
######################################

SFEM_HANDLE_RHS=0 \
SFEM_HANDLE_NEUMANN=0 \
SFEM_HANDLE_DIRICHLET=1 \
SFEM_DIRICHLET_NODES=$dirichlet_nodes \
assemble $mesh_path $workspace

potential=$workspace/potential.raw
solve $workspace/rowptr.raw $rhs $potential

p1_dpdx=$workspace/correction_x.raw
p1_dpdy=$workspace/correction_y.raw
p1_dpdz=$workspace/correction_z.raw

p1grads $potential $p1_dpdx $p1_dpdy $p1_dpdz

# Add correction to velocity

new_velx=$workspace/vel_x.raw
new_vely=$workspace/vel_y.raw
new_velz=$workspace/vel_z.raw

cp $velx $new_velx
cp $vely $new_vely
cp $velz $new_velz

axpy 1 $p1_dpdx $new_velx
axpy 1 $p1_dpdy $new_vely
axpy 1 $p1_dpdz $new_velz

integrate_divergence $mesh_path $new_velx $new_vely $new_velz 

raw_to_db.py $mesh_path $post_dir/post_db.vtk \
	--point_data="$workspace/vel*.raw,$potential,$rhs,$rhs_viz,$node_div,$sides,$workspace/correction*.raw,$velx,$vely,$velz"  \
	--cell_data="$cell_div,$cell_volume"

# Clean-up
rm -r $workspace
