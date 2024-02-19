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
PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
PATH=$SCRIPTPATH/../../python/sfem/sdf:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

field=field.raw
mesh=mesh
out=resampled
skinned=skinned
sdf=sdf.float32.raw
mesh_sorted=sorted
resample_target=$mesh_sorted
# resample_target=$skinned

if [[ -d "refined" ]]
then
	resample_target=refined
	echo "resample_target=$resample_target"
fi

if [[ -d "$skinned" ]] 
then
	echo "Reusing existing mesh $skinned and SDF!"
else
	create_sphere.sh 5
	sfc $mesh $mesh_sorted
	mkdir -p $skinned
	skin $mesh $skinned
	mesh_to_sdf.py $skinned $sdf --hmax=0.01 --margin=0.2
	raw_to_xdmf.py $sdf
fi

sizes=`head -3 metadata_sdf.float32.yml 			  | awk '{print $2}' | tr '\n' ' '`
origins=`head -8 metadata_sdf.float32.yml 	| tail -3 | awk '{print $2}' | tr '\n' ' '`
scaling=`head -11 metadata_sdf.float32.yml 	| tail -3 | awk '{print $2}' | tr '\n' ' '`

echo $sizes
echo $origins
echo $scaling

# n_procs=1
# n_procs=2
n_procs=15
LAUNCH="mpiexec -np $n_procs"
# LAUNCH="lldb --"

# export OMP_NUM_THREADS=8
# export OMP_PROC_BIND=true

set -x
time SFEM_INTERPOLATE=0 SFEM_READ_FP32=1 $LAUNCH grid_to_mesh $sizes $origins $scaling $sdf $resample_target $field
raw_to_db.py $resample_target out.vtk --point_data=$field
