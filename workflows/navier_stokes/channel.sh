#!/usr/bin/env bash

# domain = [0, 2.2] x [0, 0.41] \ B(0.2, 0.2, r), r = 0.05
#  
# 

# REFERENCES
# https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow.html
# https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
# https://www.featool.com/tutorial/2015/08/13/Accurate-Computational-Fluid-Dynamics-CFD-Simulations-with-FEATool
# https://www.youtube.com/watch?v=7Y6iSPPHwvU&t=3s
# https://drive.google.com/file/d/1iWXkEYfBGwlYKth_Cl9JdD2Z18TGCyDs/view?usp=sharing
# https://drive.google.com/file/d/1NOt8M3IpCNYu8mfT6VGBBrh6hYZ8G3FZ/view?usp=sharing

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../python/algebra:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

source ../sfem_config.sh

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
# create_box_2D_p2.sh 3
# rm -f $mesh/z.raw
nvars=3

# sleft=$SFEM_MESH_DIR/sidesets_aos/sleft.raw
# sright=$SFEM_MESH_DIR/sidesets_aos/sright.raw
# sbottom=$SFEM_MESH_DIR/sidesets_aos/sbottom.raw
# stop=$SFEM_MESH_DIR/sidesets_aos/stop.raw

sinlet=$SFEM_MESH_DIR/sidesets_aos/sinlet.raw
swall=$SFEM_MESH_DIR/sidesets_aos/swall.raw
soutlet=$SFEM_MESH_DIR/sidesets_aos/soutlet.raw

# Laminar case
# U=0.3
# Shedding
U=1.5
# U=10
python3 -c 'import numpy as np; idx=np.fromfile("'$sinlet'", dtype="'$py_sfem_idx_t'"); y=np.fromfile("'$SFEM_MESH_DIR'/y.raw",dtype="'$py_sfem_geom_t'"); y=y[idx]; U='$U'; fy=4*U*y*(0.41 - y)/(0.41*0.41); fy.astype("'$py_sfem_real_t'").tofile("bcvalues.raw")'

# export SFEM_VELOCITY_DIRICHLET_NODESET="$sbottom,$sbottom,$stop,$stop,$sleft,$sleft"
export SFEM_VELOCITY_DIRICHLET_NODESET="$swall,$swall,$sinlet,$sinlet"
export SFEM_VELOCITY_DIRICHLET_VALUE="0,0,path:bcvalues.raw,0"
export SFEM_VELOCITY_DIRICHLET_COMPONENT="0,1,0,1"

python3 -c "import numpy as np; np.array([210]).astype(np.int32).tofile('pbc.int32.raw')"
# export SFEM_PRESSURE_DIRICHLET_NODESET="pbc.int32.raw"

# export SFEM_PRESSURE_DIRICHLET_NODESET="$sright"
# export SFEM_PRESSURE_DIRICHLET_NODESET="$soutlet"
# export SFEM_PRESSURE_DIRICHLET_VALUE="0"
# export SFEM_PRESSURE_DIRICHLET_COMPONENT="0"

export SFEM_DT=0.0001
export SFEM_MAX_TIME=0.001
export SFEM_EXPORT_FREQUENCY=0.0001
export SFEM_RTOL=1e-14
export SFEM_MAX_IT=60
export SFEM_CFL=0.005
export SFEM_LUMPED_MASS=0
export SFEM_VERBOSE=0
export SFEM_AVG_PRESSURE_CONSTRAINT=0

export SFEM_DYNAMIC_VISCOSITY=0.001
export SFEM_MASS_DENSITY=1

# export SFEM_RESTART_FOLDER=batch1
# export SFEM_RESTART_ID=772

rm -rf out
mkdir -p out
set -x

# lldb -- taylor_hood_navier_stokes $SFEM_MESH_DIR out
taylor_hood_navier_stokes $SFEM_MESH_DIR out

nsteps=`ls out/u0.*.raw | wc -l | awk '{print $1}'`
raw_to_db.py $SFEM_MESH_DIR u.xmf  \
 --transient --time_whole_txt="out/time.txt" \
 --point_data="out/u0.*.raw,out/u1.*.raw" 
