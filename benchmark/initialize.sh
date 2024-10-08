#!/usr/bin/env bash



set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../workflows/sfem_config.sh
export PATH=$SCRIPTPATH/../:$PATH
export PATH=$SCRIPTPATH/../build/:$PATH
export PATH=$SCRIPTPATH/../bin/:$PATH

export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/..:$PATH
export PATH=$SCRIPTPATH/../python/sfem:$PATH
export PATH=$SCRIPTPATH/../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH

TOP_FOLDER=$PWD
BENCH_FOLDER=$PWD/db

if [[ -z $LAUNCH ]]
then
	LAUNCH=""
fi

# OpenMP
export OMP_PROC_BIND=true 
if [[ -z $OMP_NUM_THREADS ]]
then
	export OMP_NUM_THREADS=12
fi

mkdir -p $BENCH_FOLDER
cd $BENCH_FOLDER

# Clean-up if previous generation is there
rm -rf sphere
rm -rf cylinder

mkdir -p sphere
mkdir -p cylinder

#####################################
# SPHERE 
#####################################

echo "Sphere benchmark database"

cd sphere
export SPHERE_FOLDER=$PWD

refs=(0 1 2 3 4 5)
n_refs=${#refs[@]}
largest_matrix=4

for r in ${refs[@]}
do
	echo "spheres: generating case $r"
	mkdir $r
	cd $r
	mkdir -p p1 p2
	
	# P1 folder
	cd p1
	create_sphere.sh $r
	sfc mesh sorted
	refine sorted refined

	# P2 folder
	mesh_p1_to_p2 sorted ../p2

	mkdir -p matrix_scalar
	echo "op: Laplacian" > matrix_scalar/meta.yaml

	mkdir -p matrix_vector
	echo "op: LinearElasticity" > matrix_vector/meta.yaml

	if [[ $r -le $largest_matrix ]]
	then
		SFEM_HANDLE_DIRICHLET=0 SFEM_HANDLE_NEUMANN=0 SFEM_HANDLE_RHS=0 assemble refined matrix_scalar
		SFEM_HANDLE_DIRICHLET=0 SFEM_HANDLE_NEUMANN=0 SFEM_HANDLE_RHS=0 assemble3 refined matrix_vector
	fi

	cd $SPHERE_FOLDER
done

function gen_by_refinement()
{
	last=$1
	next=$((last + 1))
	
	mkdir -p $next
	mkdir -p $next/p1
	mkdir -p $next/p2

	cp -r $last/p1/refined $next/p1/mesh
	sfc $next/p1/mesh $next/p1/sorted
	refine $next/p1/sorted $next/p1/refined
	mesh_p1_to_p2 $next/p1/sorted  $next/p2

	mkdir -p $last/p1/matrix_scalar
	echo "op: Laplacian" > $last/p1/matrix_scalar/meta.yaml

	mkdir -p $last/p1/matrix_vector
	echo "op: LinearElasticity" > $last/p1/matrix_vector/meta.yaml
}

cd $BENCH_FOLDER

#####################################
# CYLINDER 
#####################################

echo "Cylinder benchmark database"

cd cylinder
export CYLINDER_FOLDER=$PWD

refs=(0 1 2 3 4)
n_refs=${#refs[@]}
largest_matrix=3

for r in ${refs[@]}
do
	echo "cylinder: generating case $r"
	mkdir $r
	cd $r
	mkdir -p p1 p2
	
	# P1 folder
	cd p1
	create_cylinder.sh $r
	sfc mesh sorted
	refine sorted refined

	# P2 folder
	mesh_p1_to_p2 sorted ../p2

	mkdir -p matrix_scalar
	echo "op: Laplacian" > matrix_scalar/meta.yaml

	mkdir -p matrix_vector
	echo "op: LinearElasticity" > matrix_vector/meta.yaml

	if [[ $r -le $largest_matrix ]]
	then
		SFEM_HANDLE_DIRICHLET=0 SFEM_HANDLE_NEUMANN=0 SFEM_HANDLE_RHS=0 assemble  refined matrix_scalar
		SFEM_HANDLE_DIRICHLET=0 SFEM_HANDLE_NEUMANN=0 SFEM_HANDLE_RHS=0 assemble3 refined matrix_vector
	fi

	cd $CYLINDER_FOLDER
done

# gen_by_refinement $last

cd $BENCH_FOLDER
cd $TOP_FOLDER
