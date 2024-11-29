#!/usr/bin/env bash

mesh=/Users/patrickzulian/Desktop/cloud/owncloud_HSLU/Patrick/2023/Cases/FP70/fluid.exo
hmax=0.00004
margin=0.0002
sdf=sdf.float32.raw 
boxed=/Users/patrickzulian/Desktop/cloud/owncloud_HSLU/Patrick/2023/Cases/FP70/solid.exo

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export SFEM_SUPERIMPOSE=1
# export REUSE_SDF=1
$SCRIPTPATH/sdf.sh $mesh $hmax $margin $sdf $boxed
