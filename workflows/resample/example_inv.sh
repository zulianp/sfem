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
out=resampled
skinned=skinned
sdf=sdf.float32.raw
mesh_sorted=sorted
resample_target=$mesh_sorted

export SFEM_INTERPOLATE=0 
export SFEM_READ_FP32=1

mpirun -n 1 mesh_to_grid --mesh_folder /home/sriva/spz/sfem_test/mesh/ --nx 100 --ny 100 --nz 100 --ox -0.88 --oy -0.88 --oz -0.88 --dx 0.0176 --dy 0.0176 --dz 0.0176 --output_path /home/sriva/spz/sfem_test/text.raw

raw_to_db.py $resample_target /home/sriva/spz/sfem_test/out.vtk --point_data=/home/sriva/spz/sfem_test/field.raw  --point_data_type=float32