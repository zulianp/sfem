#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PATH=$SFEM_PATH:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/sdf:$PATH
export PATH=$SCRIPTPATH/../../workflows/mech:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

export SFEM_EXECUTION_SPACE=device
export SFEM_ELEMENT_REFINE_LEVEL=2
export SFEM_DT=1
export SFEM_T_END=40



HERE=$PWD

rm -rf domain
if [[ ! -d domain ]]
then
	mkdir -p domain
	cd domain

	MESH_FACTOR=2
	pos=32
	box_mesh.py box --cell_type=hex8 -x $((2 * MESH_FACTOR)) -y $((2 * MESH_FACTOR)) -z $((2 * MESH_FACTOR)) --width=1 --tx=$pos --ty=$pos --tz=$pos
	surf_type=quad4
	
	skin box skin_box
	raw_to_db.py skin_box skin_box.vtk

	cd $HERE
fi

xc=64
yc=64
zc=64
obs=obs


echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

rm -rf dynamic_contact

export SFEM_COARSE_OP_TYPE=MF
# export SFEM_USE_SPMG=0

$LAUNCH elastodynamics domain/box obs dirichlet.yaml domain/skin_box/sidesets dynamic_contact
raw_to_db.py dynamic_contact/mesh dynamic_contact.xdmf -p "dynamic_contact/out/disp.0.*.raw,dynamic_contact/out/disp.1.*.raw,dynamic_contact/out/disp.2.*.raw" --transient  --time_whole_txt=dynamic_contact/out/time.txt
