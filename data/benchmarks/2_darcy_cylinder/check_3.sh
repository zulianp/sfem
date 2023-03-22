#!/usr/bin/env bash

set -e
# set -x

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../../..:$PATH
PATH=$SCRIPTPATH/../../../python:$PATH
PATH=$SCRIPTPATH/../../../python/mesh:$PATH
PATH=$SCRIPTPATH/../../../workflows/divergence:$PATH

mesh_path=./mesh
workspace=`mktemp -d`

velx=$workspace/prescribed_vel_x.raw
vely=$workspace/prescribed_vel_y.raw
velz=$workspace/prescribed_vel_z.raw

python3 -c "import numpy as np; a=np.fromfile(\"$mesh_path/x.raw\", dtype=np.float32); a = a + 1; a = a * a; a.astype(np.float64).tofile(\"$velx\")"
python3 -c "import numpy as np; a=np.fromfile(\"$mesh_path/y.raw\", dtype=np.float32); a.astype(np.float64).tofile(\"$vely\")"
python3 -c "import numpy as np; a=np.fromfile(\"$mesh_path/z.raw\", dtype=np.float32); a.astype(np.float64).tofile(\"$velz\")"

logfile=output/check_3/log.txt
divergence.sh mesh $velx $vely $velz output/check_3/div > $logfile

echo "Expected divergence ~ 0"
grep "integral div" $logfile

echo "Expected outflux ~ 0"
grep "surface_outflux = " $logfile

./projection.sh $velx $vely $velz output/check_3

# Clean-up
rm -r $workspace
