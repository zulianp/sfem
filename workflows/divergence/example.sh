#!/usr/bin/env bash

set -e
set -x

mesh=/Users/patrickzulian/Desktop/code/utopia/utopia_fe/data/hydros/mesh-multi-outlet-better
vx=../pprojection/workspace/projected/vel_x.raw
vy=../pprojection/workspace/projected/vel_y.raw
vz=../pprojection/workspace/projected/vel_z.raw

./divergence.sh $mesh $vx $vy $vz
      
    
