#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../python/algebra:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

export ISOLVER_LSOLVE_PLUGIN=$CODE_DIR/utopia/utopia/build_shared/libutopia.dylib
# export ISOLVER_LSOLVE_PLUGIN=$CODE_DIR/utopia/utopia/build_shared/libutopia.so

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true

if [[ -z "$ISOLVER_LSOLVE_PLUGIN" ]]
then
	echo "Error! Please define ISOLVER_LSOLVE_PLUGIN=<path_to_plugin.dylib>"
	exit -1
fi

SFEM_MESH_DIR=mesh
create_box_2D_p2.sh 7
# create_box_2D.sh 2
# rm -f $mesh/z.raw
nvars=3

sleft=$SFEM_MESH_DIR/sidesets_aos/sleft.raw
sright=$SFEM_MESH_DIR/sidesets_aos/sright.raw
sbottom=$SFEM_MESH_DIR/sidesets_aos/sbottom.raw
stop=$SFEM_MESH_DIR/sidesets_aos/stop.raw

export SFEM_DIRICHLET_NODESET="$sleft,$stop,$sbottom"
export SFEM_DIRICHLET_VALUE="1,0,0"
export SFEM_DIRICHLET_COMPONENT="0,0,0"

export SFEM_DT=0.1
export SFEM_MAX_TIME=0.6
export SFEM_RTOL=1e-14
export SFEM_MAX_IT=4000
export SFEM_EXPORT_FREQUENCY=0.01
export SFEM_VERBOSE=0
export SFEM_DIFFUSIVITY=1

# Used for explicit integration
export SFEM_CFL=0.05;

mkdir -p out
set -x


export SFEM_IMPLICIT=0
export SFEM_LUMPED_MASS=0

rm -rf out

# lldb -- 
heat_equation $SFEM_MESH_DIR out

nsteps=`ls out/*.raw | wc -l | awk '{print $1}'`

raw_to_db.py $SFEM_MESH_DIR u.xmf  \
 --transient --n_time_steps=$nsteps \
 --point_data="out/u.*.raw" 
