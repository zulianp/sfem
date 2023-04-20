#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../workflows/divergence:$PATH
PATH=$SCRIPTPATH/../meshes:$PATH

# create_cylinder.sh 1
create_cylinder_p2.sh 0
