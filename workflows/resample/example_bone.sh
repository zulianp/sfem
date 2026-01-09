#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source $SCRIPTPATH/../../build/sfem_config.sh
export PATH=$SCRIPTPATH/../../:$PATH
export PATH=$SCRIPTPATH/../../build/:$PATH
export PATH=$SCRIPTPATH/../../bin/:$PATH

PATH=$SCRIPTPATH:$PATH
PATH=$SCRIPTPATH/../..:$PATH
PATH=$SCRIPTPATH/../../python/sfem:$PATH
PATH=$SCRIPTPATH/../../python/sfem/mesh:$PATH
PATH=$SCRIPTPATH/../../python/sfem/grid:$PATH
PATH=$SCRIPTPATH/../../python/sfem/algebra:$PATH
PATH=$SCRIPTPATH/../../python/sfem/sdf:$PATH
PATH=$SCRIPTPATH/../../data/benchmarks/meshes:$PATH

NCORES=8

# export CUDA_LAUNCH_BLOCKING=1

export OMP_PROC_BIND=true
export OMP_NUM_THREADS=$NCORES

field=field.raw
mesh=bone

export SFEM_OUT_BASE_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


# refine torus torus2
# refine torus2 torus3
# refine torus3 torus4
# mesh=${SFEM_TORUS_MESH:-bone}
# mesh=impeller_tet4

echo "Using mesh: $mesh"

out=resampled
skinned=skinned
sdf=sdf.float32.raw

# Use HEX_SIZE environment variable if defined, otherwise use default
echo "Testing SDF quality ..."
SFEM_HEX_SIZE=${SFEM_HEX_SIZE:-125}
# sdf_test.py $sdf $SFEM_HEX_SIZE

sizes=$(echo "$SFEM_HEX_SIZE $SFEM_HEX_SIZE $SFEM_HEX_SIZE")
origins=$(echo "0.0 0.0 0.0")
scaling=$(echo "1.0 1.0 1.0")

echo $sizes
echo $origins
echo $scaling

n_procs=1
# n_procs=2
# n_procs=8

Nsight_PATH="/home/simone/App/ncu/"
Nsight_OUTPUT="/home/simone/App/ncu_out/ncu_grid_to_mesh"

# Nsight_OUTPUT="/capstor/scratch/cscs/sriva/ncu_out"

# LAUNCH="mpiexec -np $n_procs "
# LAUNCH="srun -p debug  --exclusive -n $n_procs --gpus-per-task=1  "

# LAUNCH="srun -p debug  --cpu-bind=socket  --exclusive -n $n_procs --gpus-per-task=1  ./mps-wrapper.sh "

# LAUNCH="srun -p debug -n $n_procs  ncu  --set roofline --print-details body  -f --section ComputeWorkloadAnalysis -o ${Nsight_OUTPUT} "

# LAUNCH=""
# LAUNCH="srun --cpu-bind=socket  --exclusive --gpus=$n_procs  -p debug -n $n_procs  ./mps-wrapper.sh "
# LAUNCH="srun --cpu-bind=socket  --exclusive --gpus-per-task=1  -p debug -n $n_procs  ./mps-wrapper.sh "

PROFILE=0
if [[ $PROFILE -eq 1 ]]
then
LAUNCH="${Nsight_PATH}/ncu \
        --set roofline \
        --target-processes all \
        --print-details body \
        -f \
        --replay-mode application \
        --app-replay-match all \
        --section ComputeWorkloadAnalysis \
        --section InstructionStats \
        --section SourceCounters \
        --section LaunchStats \
        --kernel-name-base demangled \
        -o ${Nsight_OUTPUT} \
        mpiexec -np $n_procs "
fi

GRID_TO_MESH="grid_to_mesh"
GRID_TO_MESH="perf record -o /tmp/out.perf grid_to_mesh"

# LAUNCH="lldb --"

# export OMP_NUM_THREADS=8
# export OMP_PROC_BIND=true


export SFEM_INTERPOLATE=0
export SFEM_READ_FP32=1
export SFEM_ADJOINT=1

export SFEM_CLUSTER_SIZE=${SFEM_CLUSTER_SIZE:-32}

if [[ $SFEM_ADJOINT -eq 1 ]]
then
	echo "Running in adjoint mode"
else
	echo "Running in forward mode"
fi

if [[ $SFEM_ADJOINT -eq 1 ]]
then
	echo "Starting adjoint run with $n_procs processes ++++++++++++++++"
fi

echo "Running resampling from grid to mesh..."
echo "  Sizes:       $sizes "
echo "  Origins:     $origins "
echo "  Scaling:     $scaling "
echo "  SDF:         $sdf "
echo "  Target mesh: $mesh "
echo "  Field:       $field "

set -x
time $LAUNCH $GRID_TO_MESH $sizes $origins $scaling $sdf $mesh $field TET4 CUDA
set +x

PRECISION=float32
raw_to_db.py $mesh out.vtk --point_data=$field  --point_data_type=$PRECISION

# Function to create metadata and convert raw files to XDMF format
# Usage: process_raw_file filename
# Example: process_raw_file field_cnt
function process_raw_file() {
    local file_base=$1
    local raw_file="${file_base}.raw"
    local metadata_file="metadata_${file_base}.yml"
    
    if [[ $SFEM_ADJOINT -eq 1 && -f $raw_file ]]; then
        echo "Processing $raw_file..."
        head -11 metadata_sdf.float32.yml > $metadata_file
        echo "path: $PWD/$raw_file" >> $metadata_file
        raw_to_xdmf.py $raw_file
    fi
}

process_raw_file test_field
# process_raw_file field_cnt
# process_raw_file bit_array
# process_raw_file test_field_fun_XYZ
# process_raw_file test_field_alpha
# process_raw_file test_field_volume

if [[ -f "test_field.xdmf" && $PRECISION == "float64" ]]; then
    sed -i 's/Precision="4"/Precision="8"/' test_field.xdmf
fi



