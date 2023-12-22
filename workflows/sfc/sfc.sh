#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python/sfem:$PATH
PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH

if (($# != 2))
then
	printf "usage: $0 <mesh_path> <output_folder>\n" 1>&2
	exit -1
fi

# Volume mesh
mesh_path=$1
output_folder=$2

mkdir -p $output_folder

nbytes=`ls -la $mesh_path/i0.raw | awk '{print $5}'`
nelements=$(( $nbytes / 4 ))

workspace=`mktemp -d`

python3 -c 'import numpy as np; a = np.arange('$nelements', dtype=np.float64); a.tofile("'$workspace'/element_id.raw");'

export SFEM_EXPORT_SFC=$workspace/sfc.raw 
sfc  $mesh_path $output_folder 

python3 -c 'import numpy as np; a = np.fromfile("'$SFEM_EXPORT_SFC'", dtype=np.uint32); a.astype(np.float64).tofile("'$SFEM_EXPORT_SFC'");'
raw_to_db.py $output_folder $output_folder/mesh.vtk --cell_data=$workspace/element_id.raw,$SFEM_EXPORT_SFC


rm -r $workspace