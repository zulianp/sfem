#!/usr/bin/env bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

set -x

rm -r $SCRIPTPATH/mesh
rm -r $SCRIPTPATH/eigs
rm -r $SCRIPTPATH/surf

rm $SCRIPTPATH/*.xmf
rm $SCRIPTPATH/*.h5
