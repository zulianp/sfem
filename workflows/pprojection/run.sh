#!/usr/bin/env bash

# Methods 
# - Average pressure 0
# - Fix 1 point
# - ?

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

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true

# Remove me!
UTOPIA_EXEC=$CODE_DIR/utopia/utopia/build/utopia_exec

if [[ -z "$UTOPIA_EXEC" ]]
then
	echo "Error! Please define UTOPIA_EXEC=<path_to_utopia_exectuable>"
	exit -1
fi

# Expensive routine
# resample=1
resample=0

mkdir -p workspace
workspace="workspace"

mkdir -p output
output=output

div_stages_csv=$output/div_stages.csv
echo "stage,div" > $div_stages_csv

solve()
{
	mat=$1
	rhs=$2
	x=$3
	# mpirun $UTOPIA_EXEC -app ls_solve -A $mat -b $rhs -out $x -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 -out $sol --verbose
	mpiexec -np 8 $UTOPIA_EXEC -app ls_solve -A $mat -b $rhs -out $x -use_amg false --use_ksp -pc_type hypre -ksp_type cg -atol 1e-18 -rtol 0 -stol 1e-19 --verbose
}

surface_divergence_check()
{
	if (($# != 4))
	then
		printf "usage: $0 <name> <velx.raw> <vely.raw> <velz.raw>\n" 1>&2
		exit -1
	fi

	name_=$1
	lux_=$2
	luy_=$3
	luz_=$4

	outflux_=$workspace/outflux.raw
	surface_outflux $surf_mesh_path $lux_ $luy_ $luz_ $outflux_
	raw_to_db.py $surf_mesh_path $output/"$name_".vtk --cell_data="$outflux_"
}


divergence_check()
{
	if (($# != 4))
	then
		printf "usage: $0 <stage> <velx.raw> <vely.raw> <velz.raw>\n" 1>&2
		exit -1
	fi

	stage_=$1
	lux_=$2
	luy_=$3
	luz_=$4
	ldivu_=$workspace/div_vel.raw

	echo "-------------------------------------------"
	echo "-------------------------------------------"
	echo "Divergence check: $stage_"
	echo "-------------------------------------------"

	SFEM_VERBOSE=1 divergence $mesh_path $lux_ $luy_ $luz_ $ldivu_
	integrate_divergence $mesh_path $lux_ $luy_ $luz_

	div_measure=`python3 -c "import numpy as np; print(np.sum((np.fromfile(\"$ldivu_\")), dtype=np.float64))"`
	
	# lumped_mass_inv $mesh_path $ldivu_ $ldivu_
	raw_to_db.py $mesh_path $output/"$stage_"_divergence.vtk --point_data="$ldivu_"

	echo "$stage_,$div_measure" >> $div_stages_csv
	rm $ldivu_

	svx=$workspace/temp_svx.raw
	svy=$workspace/temp_svy.raw
	svz=$workspace/temp_svz.raw

	sgather $surface_nodes $real_type_size $lux_ $svx
	sgather $surface_nodes $real_type_size $luy_ $svy
	sgather $surface_nodes $real_type_size $luz_ $svz

	surface_divergence_check $stage_ $svx $svy $svz
	echo "-------------------------------------------"
	echo "-------------------------------------------"
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

# set -x

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

# Boundary nodes

surf_mesh_path=$workspace/skinned

mkdir -p $surf_mesh_path

skin $mesh_path $surf_mesh_path
surface_nodes=$surf_mesh_path/node_mapping.raw
parent_elements=$surf_mesh_path/parent.raw

# Split wall from inlet and outlet

boundary_wall=$workspace/wall_idx.raw

set_diff $surface_nodes $mesh_path/zd.raw $workspace/temp.raw
set_diff $workspace/temp.raw $mesh_path/on.raw $boundary_wall

################################################
# Assemble laplacian
################################################

# List dirchlet nodes
# python3 -c "import numpy as np; np.array([7699704]).astype(np.int32).tofile('dirichlet.raw')"
fix_value=0

# dirichlet_nodes=$workspace/dirichlet.raw
# mv dirichlet.raw $dirichlet_nodes

# Wall to zero
dirichlet_nodes=$boundary_wall

SFEM_HANDLE_RHS=0 \
SFEM_HANDLE_NEUMANN=0 \
SFEM_HANDLE_DIRICHLET=1 \
SFEM_DIRICHLET_NODES=$dirichlet_nodes \
assemble $mesh_path $workspace

# NEXT SHOULD BE A LOOP OVER FILES

################################################
# Transfer from grid to mesh
################################################

mkdir -p $workspace/projected

projected=$workspace/projected/
ux=$projected/vel_x.raw
uy=$projected/vel_y.raw
uz=$projected/vel_z.raw

if [[ "$resample" -eq "1" ]]
then
	SFEM_READ_FP32=1 pizzastack_to_mesh $nx $ny $nz $gridux $mesh_path $ux
	SFEM_READ_FP32=1 pizzastack_to_mesh $nx $ny $nz $griduy $mesh_path $uy
	SFEM_READ_FP32=1 pizzastack_to_mesh $nx $ny $nz $griduz $mesh_path $uz
fi

divergence_check "after_transfer" $ux $uy $uz
raw_to_db.py $mesh_path $output/projected_gradient.vtk --point_data="$projected/vel_*.raw"


################################################
# Set velocity to zero on surface nodes
################################################

mux=$workspace/mux.raw
muy=$workspace/muy.raw
muz=$workspace/muz.raw

# smask $boundary_wall $ux $mux 0
# smask $boundary_wall $uy $muy 0
# smask $boundary_wall $uz $muz 0

cp $ux $mux
cp $uy $muy
cp $uz $muz

divergence_check "after_clamping" $mux $muy $muz

################################################
# Compute (div(u), test)_L2
################################################

divu=$workspace/divu.raw
divergence $mesh_path $mux $muy $muz $divu

################################################
# Solve linear system
################################################

# Fix degree of freedom for uniqueness of solution
rhs=$workspace/rhs.raw
smask $dirichlet_nodes $divu $rhs $fix_value
# smask $surface_nodes $divu $rhs $fix_value

potential=$workspace/potential.raw
solve $workspace/rowptr.raw $rhs $potential

raw_to_db.py $mesh_path $output/potential.vtk --point_data=$potential

################################################
# Compute gradients
################################################

p1_dpdx=$workspace/vel_x.raw
p1_dpdy=$workspace/vel_y.raw
p1_dpdz=$workspace/vel_z.raw
p1grads $potential $p1_dpdx $p1_dpdy $p1_dpdz

raw_to_db.py $mesh_path $output/gradient_corrector.vtk --point_data="$workspace/vel_*.raw"

# Add correction to velocity
axpy 1 $mux $p1_dpdx 
axpy 1 $muy $p1_dpdy 
axpy 1 $muz $p1_dpdz 

divergence_check "after_correction" $p1_dpdx $p1_dpdy $p1_dpdz

# Show corrected velocities
raw_to_db.py $mesh_path $output/corrected_gradient.vtk --point_data="$workspace/vel_*.raw"

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

raw_to_db.py $surf_mesh_path $output/wssmag.vtk --cell_data="$wssmag,$sshear_prefix.*.raw"
