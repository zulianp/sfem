#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

set -x

rsync -av $SCRIPTPATH/..  alps:/capstor/scratch/cscs/zulianp/sfem_simone/sfem --exclude 'api' --exclude 'venv' --exclude 'build*' --exclude '*.o' --exclude '.DS_Store' --exclude '*git' --exclude 'benchmark/db' 