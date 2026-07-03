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

export DYLD_LIBRARY_PATH=$CODE_DIR/external/petsc/lib/:$CODE_DIR/utopia/utopia/build/ui/
export ISOLVER_LSOLVE_PLUGIN=$INSTALL_DIR/utopia/lib/libutopia.dylib
export UTOPIA_LINEAR_SOLVER_CONFIG=$PWD/utopia.yaml

echo "ssolve driver removed (ISolver dependency dropped from SFEM)." >&2
echo "Use utopia or another SFEM solver workflow for this benchmark." >&2
exit 1
