#!/usr/bin/env bash

set -e

# Uncomment to activate virtual environment if needed
# source $CODE_DIR/merge_git_repos/sfem/venv/bin/activate

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	echo "Example: export SFEM_PATH=/users/hyang/ws/sfem_github/sfem/build"
	exit 1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PATH=$SFEM_PATH/bin:$PATH
export PATH=$SFEM_PATH/scripts/sfem/mesh/:$PATH
export PATH=$SFEM_PATH/scripts/sfem/grid/:$PATH
export PATH=$SFEM_PATH/scripts/sfem/sdf/:$PATH
export PATH=$SFEM_PATH/workflows/mech/:$PATH
export PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

HERE=$PWD

# Create geometry if it doesn't exist
rm -rf torus_geometry
if [[ ! -d torus_geometry ]]
then
	mkdir -p torus_geometry
	cd torus_geometry

	# Convert VTK file to raw format
	db_to_raw.py ../torus_cut_hex.vtk torus
	surf_type=quad4
	
	# Extract surface mesh
	skin torus skin_torus
	raw_to_db.py skin_torus skin_torus.vtk

	set -x

	# Create inlet and outlet sidesets
	SFEM_DEBUG=1 create_sideset torus   4.96663 0.0306772 1.18258  0.998 	inlet
	SFEM_DEBUG=1 create_sideset torus   4.98568 9.98184   1.10985  0.998 	outlet

	# Visualize boundaries
	raw_to_db.py inlet/surf 			inlet/surf.vtk 				--coords=torus --cell_type=$surf_type
	raw_to_db.py outlet/surf 			outlet/surf.vtk 			--coords=torus --cell_type=$surf_type
	raw_to_db.py torus 					torus.vtk 

	cd $HERE
fi

echo "======================================"
echo "Kelvin-Voigt Newmark Torus Test"
echo "======================================"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

# Material parameters
# k: shear modulus (mu)
# K: bulk modulus
# eta: viscosity coefficient
# rho: density
export SFEM_K=4.0
export SFEM_BULK_MODULUS=3.0
export SFEM_ETA=0.1
export SFEM_RHO=1.0

# Newmark time integration parameters
export SFEM_DT=0.01           # time step size
export SFEM_T_END=1.0         # end time
export SFEM_GAMMA=0.5         # Newmark gamma parameter
export SFEM_BETA=0.25         # Newmark beta parameter

# Output settings
export SFEM_NEWMARK_ENABLE_OUTPUT=1
export SFEM_EXPORT_FREQUENCY=10  # export every 10 steps

# Use GPU if available
# export SFEM_EXECUTION_SPACE=device

# Clean previous output
rm -rf torus_kv_output

# Run Kelvin-Voigt Newmark test
# Note: Verify the actual executable name
# It might be sfem_NewmarkKVTest or similar
echo "Running Kelvin-Voigt Newmark simulation..."

# Option 1: If there's a dedicated driver program
# $LAUNCH kv_newmark torus_geometry/torus dirichlet_torus_kv.yaml torus_kv_output

# Option 2: Use test program
$LAUNCH sfem_NewmarkKVTest

# Convert output to VTK format for visualization if output exists
if [[ -d torus_kv_output ]]
then
	echo "Converting output to VTK..."
	raw_to_db.py torus_kv_output/mesh torus_kv_output.vtk \
		-p 'torus_kv_output/disp/*.raw' \
		-p 'torus_kv_output/velocity/*.raw' \
		-p 'torus_kv_output/acceleration/*.raw' \
		$EXTRA_OPTIONS
	echo "Output saved to torus_kv_output.vtk"
fi

echo "======================================"
echo "Simulation completed!"
echo "======================================"

