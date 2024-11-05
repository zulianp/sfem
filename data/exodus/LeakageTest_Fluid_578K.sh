#!/usr/bin/env bash
set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
$SCRIPTPATH/LeakageTest_Fluid.sh $SCRIPTPATH/LeakageTest_Fluid_578K.exo
