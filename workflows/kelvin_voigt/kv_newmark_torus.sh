#!/usr/bin/env bash

set -e

# Load CUDA on daint
module load cudatoolkit 2>/dev/null || module load cuda 2>/dev/null || true

export SFEM_PATH=/users/hyang/ws/sfem_github/sfem/build_release
if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Python/venv not needed - mesh is pre-prepared
# source /users/hyang/ws/sfem_github/sfem/venv/bin/activate
# export PYTHONPATH=/users/hyang/ws/sfem_github/sfem/venv/lib/python3.12/site-packages:$PYTHONPATH

export PATH=$SFEM_PATH:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
export PATH=$SCRIPTPATH/../../python/sfem/sdf:$PATH
export PATH=$SCRIPTPATH/../../workflows/mech:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH



export SFEM_EXECUTION_SPACE=device
export SFEM_ELEMENT_REFINE_LEVEL=2
export SFEM_DT=0.15
export SFEM_T_END=15




HERE=$PWD

# Skip mesh generation - use pre-prepared mesh from hyperelasticity folder
# Copy with: cp -r ../hyperelasticity/torus_geometry .
if [[ ! -d torus_geometry/torus ]]
then
	echo "ERROR: torus_geometry/torus not found!"
	echo "Please copy from hyperelasticity: cp -r ../hyperelasticity/torus_geometry ."
	exit 1
fi

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

rm -rf torus_kv_output


# Use SIDESET-based Neumann input (expects parent.raw in the given directory)
export SFEM_NEUMANN_SIDESET=torus_geometry/outlet
export SFEM_NEUMANN_COMPONENT=1
export SFEM_NEUMANN_VALUE=-5


$LAUNCH kelvin_voigt_newmark torus_geometry/torus dirichlet_torus_kv.yaml torus_kv_output neumann_torus_kv.yaml

# Convert output to VTK (requires Python - run on login node if needed)
# raw_to_db.py torus_kv_output/mesh torus_kv_output.vtk -p 'torus_kv_output/out/*.raw' $EXTRA_OPTIONS

