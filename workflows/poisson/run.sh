#!/usr/bin/env bash

set -e
set -x

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH

# export DYLD_LIBRARY_PATH=$INSTALL_DIR/ginkgo/lib
# export ISOLVER_LSOLVE_PLUGIN=$INSTALL_DIR/isolver/lib/libisolver_ginkgo.dylib 

export DYLD_LIBRARY_PATH=$CODE_DIR/external/petsc/lib/:/Users/patrickzulian/Desktop/code/utopia/utopia/build/ui/
export ISOLVER_LSOLVE_PLUGIN=$INSTALL_DIR/utopia/lib/libutopia.dylib
export UTOPIA_LINEAR_SOLVER_CONFIG=$PWD/utopia.yaml

# Parameters
MESH_PATH=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better
DIRICHLET_PATH=$MESH_PATH/zd.raw
NEUMANN_PATH=$MESH_PATH/on.raw
OUTPUT_PATH=$PWD/out.raw
SIMULATION_DATE=`date`

rm -f sim.yaml temp.yaml
( echo "cat <<EOF >sim.yaml";
  cat template.yaml;
  echo "EOF";
) >temp.yaml
. temp.yaml

rm temp.yaml
cat sim.yaml

ssolve sim.yaml

raw2mesh.py -d $MESH_PATH --field=$OUTPUT_PATH --field_dtype=float64 --output=out.vtk
