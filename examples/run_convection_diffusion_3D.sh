#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../workflows/sfem_config.sh
export PATH=$SCRIPTPATH/../:$PATH
export PATH=$SCRIPTPATH/../build/:$PATH
export PATH=$SCRIPTPATH/../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../python/sfem:$PATH
PATH=$SCRIPTPATH/../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../python/sfem/utils:$PATH
PATH=$SCRIPTPATH/../python/sfem/algebra:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH

HERE=$PWD

mkdir -p convection_diffusion
rm -rf convection_diffusion/out

cd convection_diffusion

# export OMP_NUM_THREADS=16
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true

SFEM_MESH_DIR=mesh
export SFEM_ELEM_TYPE=tetra

rm -rf $SFEM_MESH_DIR
create_box.sh $SFEM_ELEM_TYPE 50 50 100 1 1 2 $SFEM_MESH_DIR

export SFEM_MAX_TIME=2
export SFEM_DT=0.001
export SFEM_EXPORT_FREQUENCY=0.01
export SFEM_DIFFUSIVITY=0.1
export SFEM_VELX=0
export SFEM_VELY=0
export SFEM_VELZ=1
export SFEM_INITIAL_CONDITION="ivp.raw"

eval_nodal_function.py \
	"(1-x)*x *(1-y)*y * (2-z)**4" \
	$SFEM_MESH_DIR/x.raw $SFEM_MESH_DIR/y.raw $SFEM_MESH_DIR/z.raw $SFEM_INITIAL_CONDITION

run_convection_diffusion $SFEM_MESH_DIR out

raw_to_db.py $SFEM_MESH_DIR out.xmf  \
 --transient \
 --point_data="out/c.*.raw" \
 --time_whole_txt="out/time.txt" \
 --cell_type=$SFEM_ELEM_TYPE

raw_to_db.py $SFEM_MESH_DIR extras.vtk  \
 --point_data="out/cv_volumes.raw,out/lapl_one.float64.raw" \
 --cell_type=$SFEM_ELEM_TYPE

cd $HERE
