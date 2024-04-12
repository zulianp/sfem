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
# export SFEM_ELEM_TYPE=quad 
export SFEM_ELEM_TYPE=triangle

rm -rf $SFEM_MESH_DIR
create_box_2D.sh 1 2 1

sleft=$SFEM_MESH_DIR/sidesets_aos/sleft.raw
sright=$SFEM_MESH_DIR/sidesets_aos/sright.raw

# export SFEM_DIRICHLET_NODESET="$sleft"
# export SFEM_DIRICHLET_VALUE="1"
# export SFEM_DIRICHLET_COMPONENT="0"

export SFEM_MAX_TIME=2
export SFEM_DT=0.0001
export SFEM_EXPORT_FREQUENCY=0.02
export SFEM_DIFFUSIVITY=0
export SFEM_VELX=1
export SFEM_VELY=0
export SFEM_INITIAL_CONDITION="ivp.raw"

eval_nodal_function.py "np.exp(-400*(0.5-y)**2) * np.exp(-400*(0.2-x)**2)" $SFEM_MESH_DIR/x.raw $SFEM_MESH_DIR/y.raw $SFEM_MESH_DIR/y.raw $SFEM_INITIAL_CONDITION

# lldb -- 
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
