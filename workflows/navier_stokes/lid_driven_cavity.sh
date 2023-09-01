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
create_box_2D_p2.sh 5
rm -f $mesh/z.raw
nvars=3

sleft=$SFEM_MESH_DIR/sidesets_aos/sleft.raw
sright=$SFEM_MESH_DIR/sidesets_aos/sright.raw
sbottom=$SFEM_MESH_DIR/sidesets_aos/sbottom.raw
stop=$SFEM_MESH_DIR/sidesets_aos/stop.raw

export SFEM_VELOCITY_DIRICHLET_NODESET="$sleft,$sleft,$sright,$sright,$sbottom,$sbottom,$stop,$stop"
export SFEM_VELOCITY_DIRICHLET_VALUE="0,0,0,0,0,0,1,0,"
export SFEM_VELOCITY_DIRICHLET_COMPONENT="0,1,0,1,0,1,0,1"

python3 -c "import numpy as np; np.array([0]).astype(np.int32).tofile('pbc.int32.raw')"

export SFEM_PRESSURE_DIRICHLET_NODESET="pbc.int32.raw"
export SFEM_PRESSURE_DIRICHLET_VALUE="0"
export SFEM_PRESSURE_DIRICHLET_COMPONENT="0"

export SFEM_DT=0.000001
export SFEM_MAX_TIME=0.1
export SFEM_EXPORT_FREQUENCY=0.001
export SFEM_RTOL=1e-14
export SFEM_MAX_IT=2000
export SFEM_CFL=0.05
export SFEM_LUMPED_MASS=1
export SFEM_VERBOSE=1

export SFEM_DYNAMIC_VISCOSITY=1
export SFEM_MASS_DENSITY=1

rm -rf out
mkdir -p out
set -x

# lldb -- 
taylor_hood_navier_stokes $SFEM_MESH_DIR out

nsteps=`ls out/u0.*.raw | wc -l | awk '{print $1}'`

raw_to_db.py $SFEM_MESH_DIR u.xmf  \
 --transient --n_time_steps=$nsteps \
 --point_data="out/u0.*.raw,out/u1.*.raw" 
