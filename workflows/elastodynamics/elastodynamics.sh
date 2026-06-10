#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH

export SFEM_EXECUTION_SPACE=device
export SFEM_ELEMENT_REFINE_LEVEL=8
export SFEM_DT=0.25
export SFEM_T_END=9.75
export SFEM_ATOL=1e-7

HERE=$PWD

rm -rf domain
if [[ ! -d domain ]]
then
	mkdir -p domain
	cd domain

	MESH_FACTOR=2
	cube HEX8 $((2 * MESH_FACTOR)) $((2 * MESH_FACTOR)) $((2 * MESH_FACTOR)) 21 21 21 61 61 61 box
	surf_type=quad4
	
	skin box skin_box
	raw_to_db skin_box skin_box.vtk

	cd $HERE
fi

xc=64
yc=64
zc=64
obs=obs
out=dynamic_contact/out

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

rm -rf dynamic_contact

export SFEM_COARSE_OP_TYPE=MF

$LAUNCH elastodynamics domain/box obs dirichlet.yaml domain/skin_box/parent_sideset dynamic_contact
raw_to_db dynamic_contact/mesh dynamic_contact.xdmf -p "$out/disp.0.*.*,$out/disp.1.*.*,$out/disp.2.*.*,$out/velocity.0.*.*,$out/velocity.1.*.*,$out/velocity.2.*.*,$out/acceleration.0.*.*,$out/acceleration.1.*.*,$out/acceleration.2.*.*,$out/gap.0.*.*,$out/gap.1.*.*,$out/gap.2.*.*" --transient  --time_whole_txt=dynamic_contact/out/time.txt
