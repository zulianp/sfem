#!/usr/bin/env bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

set -e

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../data/benchmarks/meshes:$PATH
PATH=$SCRIPTPATH/../../../../matrix.io:$PATH
PATH=$SCRIPTPATH/../../convert_exodus_to_raw:$PATH


export ISOLVER_LSOLVE_PLUGIN=$CODE_DIR/utopia/utopia/build_shared/libutopia.dylib
# export ISOLVER_LSOLVE_PLUGIN=$CODE_DIR/utopia/utopia/build_shared/libutopia.so

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true

if [[ -z "$ISOLVER_LSOLVE_PLUGIN" ]]
then
	echo "Error! Please define ISOLVER_LSOLVE_PLUGIN=<path_to_plugin.dylib>"
	exit -1
fi


export VAR_UX=0
export VAR_UY=1
export VAR_UZ=2


export SFEM_VISCOSITY=""
export SFEM_MASS_DENSITY="1"
export SFEM_OUTPUT_DIR=sfem_output

##########################################################################################

export SFEM_MESH_DIR=/Users/patrickzulian/Desktop/code/sfem/data/benchmarks/5_channel/mesh

sinlet=$SFEM_MESH_DIR/sidesets_aos/sinlet.raw
soutlet=$SFEM_MESH_DIR/sidesets_aos/soutlet.raw
swall=$SFEM_MESH_DIR/sidesets_aos/swall.raw

export SFEM_VELOCITY_DIRICHLET_NODESET="$sinlet,$sinlet,$sinlet,$swall,$swall,$swall"
export SFEM_VELOCITY_DIRICHLET_VALUE="0.1,0,0,0,0,0"
export SFEM_VELOCITY_DIRICHLET_COMPONENT="$VAR_UX,$VAR_UY,$VAR_UZ,$VAR_UX,$VAR_UY,$VAR_UZ"

export SFEM_PRESSURE_DIRICHLET_NODESET="$soutlet"
export SFEM_PRESSURE_DIRICHLET_VALUE="0"
export SFEM_PRESSURE_DIRICHLET_COMPONENT="0"

export SFEM_DT=0.0000001
# export SFEM_MAX_TIME=0.00000002
export SFEM_MAX_TIME=0.0000001
export SFEM_EXPORT_FREQUENCY=$SFEM_DT
export SFEM_RTOL=1e-14
export SFEM_MAX_IT=2000
export SFEM_CFL=0.5
export SFEM_LUMPED_MASS=1
export SFEM_VERBOSE=1

export SFEM_VISCOSITY=1
export SFEM_MASS_DENSITY=1

rm -rf out
mkdir -p out
set -x

# lldb -- 
taylor_hood_navier_stokes $SFEM_MESH_DIR out

nsteps=`ls out/u0.*.raw | wc -l | awk '{print $1}'`

raw_to_db.py $SFEM_MESH_DIR u.xmf  \
 --transient --n_time_steps=$nsteps \
 --point_data="out/u0.*.raw,out/u1.*.raw,out/u2.*.raw" 

raw_to_db.py $SFEM_MESH_DIR/p1 p.xmf \
 --transient --n_time_steps=$nsteps \
 --point_data="out/p.*.raw" 
