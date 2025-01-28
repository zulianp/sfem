#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

set -x

rsync -av $SCRIPTPATH/..  zulianp@rosa.usi.ch:/scratch/zulianp/sfem --exclude 'api' --exclude 'venv' --exclude 'build*' --exclude '*.o' --exclude '.DS_Store' --exclude '*git' --exclude 'benchmark/db' #--exclude '*.raw' --exclude '*.vtk'
rsync -av $SCRIPTPATH/../../matrix.io/  zulianp@rosa.usi.ch:/scratch/zulianp/matrix.io --exclude 'api' --exclude 'venv' --exclude 'build*' --exclude '*.o' --exclude '.DS_Store' --exclude '*git' --exclude 'benchmark/db' #--exclude '*.raw' --exclude '*.vtk'