#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

set -x

rsync -av $SCRIPTPATH/..  daint:/scratch/snx3000/zulianp/sfem_runs/sfem --exclude 'api' --exclude 'venv' --exclude 'build' --exclude '*.o' --exclude '.DS_Store' --exclude '*git' --exclude 'benchmark/db' --exclude '*.raw' --exclude '*.vtk'
