#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

set -x

rsync -av $SCRIPTPATH/../data/vtk/  daint:/scratch/snx3000/zulianp/sfem_runs/sfem/data/vtk --include="*.vtk" --exclude="*.raw"
