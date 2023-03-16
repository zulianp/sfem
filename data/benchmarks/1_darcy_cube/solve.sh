#!/usr/bin/env bash

set -e
# set -x

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH

mkdir -p output

set -x

# export DYLD_LIBRARY_PATH=$INSTALL_DIR/ginkgo/lib:$DYLD_LIBRARY_PATH
# export ISOLVER_LSOLVE_PLUGIN=$INSTALL_DIR/isolver/lib/libisolver_ginkgo.dylib

export DYLD_LIBRARY_PATH=$CODE_DIR/external/petsc/lib/:/Users/patrickzulian/Desktop/code/utopia/utopia/build/ui/
export ISOLVER_LSOLVE_PLUGIN=$INSTALL_DIR/utopia/lib/libutopia.dylib
# export UTOPIA_LINEAR_SOLVER_CONFIG=$PWD/utopia.yaml

ssolve config.yaml

raw_to_db.py mesh output/db.vtk --point_data="./output/*.raw"

