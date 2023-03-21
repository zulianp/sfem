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
real_type_size=8

p1lapl()
{
	if (($# != 2))
	then
		printf "usage: $0 <potential.raw> <out.raw>\n" 1>&2
		exit -1
	fi

	potential=$1
	out=$2

	lapl $mesh_path $potential $out
	# lumped_mass_inv $mesh_path $out $out
}

p1grads()
{
	if (($# != 4))
	then
		printf "usage: $0 <potential.raw> <outx.raw> <outy.raw> <outz.raw>\n" 1>&2
		exit -1
	fi

	potential=$1
	p1_dpdx=$2
	p1_dpdy=$3
	p1_dpdz=$4

	# Per Cell quantities
	p0_dpdx=$workspace/temp_p0_dpdx.raw
	p0_dpdy=$workspace/temp_p0_dpdy.raw
	p0_dpdz=$workspace/temp_p0_dpdz.raw

	# coefficients: P1 -> P0
	SFEM_SCALE=-1 cgrad $mesh_path $potential $p0_dpdx $p0_dpdy $p0_dpdz

	################################################
	# P0 to P1 projection
	################################################

	# coefficients: P0 -> P1
	projection_p0_to_p1 $mesh_path $p0_dpdx $p1_dpdx
	projection_p0_to_p1 $mesh_path $p0_dpdy $p1_dpdy
	projection_p0_to_p1 $mesh_path $p0_dpdz $p1_dpdz
}

post_dir=output/post
mkdir -p $post_dir

# Create zero
velx=$workspace/vel_x.raw
vely=$workspace/vel_y.raw
velz=$workspace/vel_z.raw

python3 -c "import numpy as np; a=np.fromfile(\"$mesh_path/x.raw\", dtype=np.float32); a.fill(0); a.astype(np.float64).tofile(\"$velx\")"
python3 -c "import numpy as np; a=np.fromfile(\"$mesh_path/y.raw\", dtype=np.float32); a.astype(np.float64).tofile(\"$vely\")"
python3 -c "import numpy as np; a=np.fromfile(\"$mesh_path/z.raw\", dtype=np.float32); a.astype(np.float64).tofile(\"$velz\")"

ls -la $velx
ls -la $vely
ls -la $velz

# Compute gradients!
# p1grads output/potential.raw $post_dir/vel_x.raw $post_dir/vel_y.raw $post_dir/vel_z.raw
# p1lapl  output/potential.raw $post_dir/lapl.raw

raw_to_db.py mesh $post_dir/post_db.vtk --point_data="$workspace/vel*.raw"
 
divergence.sh mesh $velx $vely $velz output/div

# Clean-up
rm -r $workspace
