#!/usr/bin/env bash

set -e

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

python3 -c "import numpy as np; a=np.fromfile(\"$mesh_path/x.raw\", dtype=np.float32); a.fill(1); a.astype(np.float64).tofile(\"$velx\")"
python3 -c "import numpy as np; a=np.fromfile(\"$mesh_path/y.raw\", dtype=np.float32); a.astype(np.float64).tofile(\"$vely\")"
python3 -c "import numpy as np; a=np.fromfile(\"$mesh_path/z.raw\", dtype=np.float32); a.astype(np.float64).tofile(\"$velz\")"

ls -la $velx
ls -la $vely
ls -la $velz

post_dir=output/check_0
mkdir -p $post_dir




# logfile=$post_dir/log.txt
# divergence.sh mesh $velx $vely $velz $post_dir/div > $logfile
# divergence.sh mesh $velx $vely $velz $post_dir/div 

node=`find_closest_point.py -1 0 0 $mesh_path/x.raw $mesh_path/y.raw $mesh_path/z.raw`


# cat $mesh_path/i4.raw >  p2idx.raw
# cat $mesh_path/i5.raw >> p2idx.raw
# cat $mesh_path/i6.raw >> p2idx.raw

# cat $mesh_path/i7.raw >> p2idx.raw
# cat $mesh_path/i8.raw >> p2idx.raw
# cat $mesh_path/i9.raw >> p2idx.raw

# node=`sparse_find_closest_point.py -1 0 0 $mesh_path/x.raw $mesh_path/y.raw $mesh_path/z.raw p2idx.raw`


export dirichlet_nodes=$workspace/dirichlet_nodes.raw
python3 -c "import numpy as np; a = np.array([$node]); a.astype(np.int32).tofile(\"$dirichlet_nodes\"); print(f'fixed node = {a}');"


# python3 -c "import numpy as np; b = np.fromfile(\"$mesh_path/surface/wall/i0.raw\",dtype=np.int32); a = np.array(b[0]); a.astype(np.int32).tofile(\"$dirichlet_nodes\"); print(f'fixed node = {a}');"
# ./projection.sh $velx $vely $velz $post_dir

# python3 -c "import numpy as np; b = np.fromfile(\"$mesh_path/surface/inlet/i0.raw\",dtype=np.int32); a = np.array(b[0]); a.astype(np.int32).tofile(\"$dirichlet_nodes\"); print(f'fixed node = {a}');"

./projection_2.sh $velx $vely $velz $post_dir

# Clean-up
rm -r $workspace
