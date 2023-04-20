#!/usr/bin/env bash

source ../sfem_config.sh

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/..:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

# create_cylinder.sh 0
create_cylinder_p2.sh 0
mesh=mesh

mkdir -p workspace

# set -x
mesh_evalf.py $mesh/x.raw $mesh/y.raw $mesh/z.raw 'x*x + y*y + z*z' workspace/fx.raw

ls -la $mesh/*.raw
ls -la workspace/fx.raw

grad.sh $mesh workspace/fx.raw workspace
