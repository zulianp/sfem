#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../../build/sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

export PATH=$SCRIPTPATH:$PATH
export PATH=$SCRIPTPATH/../..:$PATH
export PATH=$SCRIPTPATH/../../python/sfem:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/utils:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true 

coarse=mesh
fine=fine

if [[ -d "$fine" ]]
then
	echo "Reusing $simulation_coarse"
else
	create_sphere.sh 3
	mesh_p1_to_p2 $coarse $fine
fi

mkdir -p fields
eval_nodal_function.py "x" $coarse/x.raw $coarse/y.raw  $coarse/z.raw fields/x.raw

# usage: %s <mesh> <from_element> <to_element> <input.float64> <output.float64>
hierarchical_prolongation $fine "TET4" 		 "MACRO_TET4" fields/x.raw 		fields/fine_x.raw
hierarchical_restriction  $fine "MACRO_TET4" "TET4"  	  fields/fine_x.raw fields/coarse_x.raw 

raw_to_db.py $fine   fine.vtk   --point_data="fields/fine_x.raw"
raw_to_db.py $coarse coarse.vtk --point_data="fields/x.raw,fields/coarse_x.raw"
