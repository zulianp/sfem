#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../../build/sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python/sfem:$PATH
PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
PATH=$SCRIPTPATH/../../python/sfem/sdf:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

field=field.raw
mesh=mesh
p2_mesh=p2_mesh
out=resampled
skinned=skinned
sdf=sdf.float32.raw
mesh_sorted=sorted
resample_target=$p2_mesh

if [[ -d "refined" ]]
then
	resample_target=refined
	echo "resample_target=$resample_target"
fi

if [[ -d "$p2_mesh" ]] 
then
	echo "Reusing existing mesh $p2_mesh!"
else
	create_sphere.sh 3
	# create_sphere.sh 0 # Visibily see the curvy surface
	sfc $mesh $mesh_sorted
	# Project p2 nodes to sphere isosurfaces (to check if nonlinear map are creating errors)
	SFEM_SPERE_TOL=1e-5 SFEM_MAP_TO_SPHERE=1 mesh_p1_to_p2 $mesh_sorted $p2_mesh

	raw_to_db.py $p2_mesh test_mapping.vtk 
fi

if [[ -f "$sdf" ]]
then
	echo "Reusing existing sdf $sdf!"
else
	echo "Computing SDF!"
	mkdir -p $skinned
	skin $mesh_sorted $skinned
	mesh_to_sdf.py $skinned $sdf --hmax=0.01 --margin=0.1
	raw_to_xdmf.py $sdf
fi

sizes=`head -3 metadata_sdf.float32.yml 			  | awk '{print $2}' | tr '\n' ' '`
origins=`head -8 metadata_sdf.float32.yml 	| tail -3 | awk '{print $2}' | tr '\n' ' '`
scaling=`head -11 metadata_sdf.float32.yml 	| tail -3 | awk '{print $2}' | tr '\n' ' '`

echo $sizes
echo $origins
echo $scaling


# export OMP_PROC_BIND=true
# export OMP_NUM_THREADS=8

# n_procs=18
n_procs=8
# n_procs=2
# n_procs=1

if [[ -z "$LAUNCH" ]]
then
	LAUNCH="mpiexec -np $n_procs"
fi

GRID_TO_MESH="grid_to_mesh"

# To enable iso-parametric transformation of p2 meshes
# for the resampling

# Enable second order mesh parametrizations
export SFEM_ENABLE_ISOPARAMETRIC=1

set -x
time SFEM_INTERPOLATE=0 SFEM_READ_FP32=1 $LAUNCH $GRID_TO_MESH $sizes $origins $scaling $sdf $resample_target $field
raw_to_db.py $resample_target out.vtk --point_data=$field
