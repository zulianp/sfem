#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../workflows/divergence:$PATH
PATH=$SCRIPTPATH/../meshes:$PATH

rm -rf mesh
create_cylinder.sh 3
# create_cylinder_p2.sh 3
