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
PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

UTOPIA_EXEC=$CODE_DIR/utopia/utopia/build/utopia_exec

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true

if [[ -z "$UTOPIA_EXEC" ]]
then
	echo "Error! Please define UTOPIA_EXEC=<path_to_utopia_exectuable>"
	exit -1
fi

solve()
{
	mat_=$1
	rhs_=$2
	x_=$3

	echo "rhs=$rhs_"
	mpiexec -np 8 $UTOPIA_EXEC -app ls_solve -A $mat_ -b $rhs_ -out $x_ -use_amg false --use_ksp -pc_type lu -ksp_type preonly
	# mpiexec -np 8 $UTOPIA_EXEC -app ls_solve -A $mat_ -b $rhs_ -out $x_ -use_amg false --use_ksp -pc_type bjacobi -ksp_type bicg --verbose
}

mesh=mesh
# create_square.sh 4
# rm -f $mesh/z.raw
nvars=3

export SFEM_DIRICHLET_NODES=all.raw
cat $mesh/sidesets_aos/*.raw > $SFEM_DIRICHLET_NODES


export SFEM_PROBLEM_TYPE=3
# export SFEM_AOS=1

if [[ -z "$SFEM_AOS" ]]
then
	stokes $mesh stokes_block_system
	crs.py $nvars $nvars stokes_block_system stokes_system

	blocks.py 'stokes_block_system/rhs.*.raw' stokes_system/rhs.raw

	solve stokes_system/rowptr.raw stokes_system/rhs.raw x.raw

	unblocks.py $nvars x.raw
	unblocks.py $nvars stokes_system/rhs.raw

	# <mesh> <ux.raw> <uy.raw> <uz.raw> <p.raw>
	stokes_check $mesh ref_vel_x.raw ref_vel_y.raw ref_p.raw

	raw_to_db.py $mesh out.vtk --point_data="x.*.raw,stokes_system/rhs.*.raw,ref_*"
else
	mkdir -p stokes_system_aos
	mkdir -p out
	set -x
	
	# lldb -- 
	stokes $mesh stokes_system_aos

	solve stokes_system_aos/rowptr.raw stokes_system_aos/rhs.raw out/x.raw
	aos_to_soa out/x.raw 8 $nvars ./out/x
	raw_to_db.py $mesh out.vtk --point_data="out/x.*.raw"
fi
