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
PATH=$SCRIPTPATH/../python/sfem/algebra:$PATH
PATH=$SCRIPTPATH/../data/benchmarks/meshes:$PATH

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true

SFEM_MESH_DIR=mesh
# create_cylinder.sh 5

sleft=$SFEM_MESH_DIR/sidesets_aos/sinlet.raw
sright=$SFEM_MESH_DIR/sidesets_aos/soutlet.raw

export SFEM_DIRICHLET_NODESET="$sleft,$sright"
export SFEM_DIRICHLET_VALUE="0,0.5"
export SFEM_DIRICHLET_COMPONENT="0,0"

run_poisson $SFEM_MESH_DIR out.raw

raw_to_db.py $SFEM_MESH_DIR x.vtk -p "out.raw"