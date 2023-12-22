#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

SFEM_DIR=$SCRIPTPATH/../..
export PATH=$SFEM_DIR:$PATH
export PATH=$SFEM_DIR/python/sfem:$PATH
export PATH=$SFEM_DIR/python/sfem/mesh:$PATH

mesh=/Users/patrickzulian/Desktop/cloud/owncloud_HSLU/Patrick/2023/Cases/FP70/solid.exo

db_to_raw.py $mesh db_raw

./sfc.sh db_raw db_raw_sfc

raw_to_db.py db_raw_sfc sfc.vtk
