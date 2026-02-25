#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=sfem-gpu-%j
#SBATCH --partition=debug
#SBATCH --time=00:03:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --output=sfem_%x_%j.out
#SBATCH --error=sfem_%x_%j.err

set -euo pipefail

DELETE_LOG="${DELETE_LOG:-0}"
IS_DEBUG="${IS_DEBUG:-0}"
BUILD_DIR="${BUILD_DIR:-$HOME/ws/sfem_github/sfem/build_debug}"
BIN_NAME="${BIN_NAME:-sfem_NewmarkKVTest}"
BIN_ARGS="${BIN_ARGS:-}"
IS_CMAKE_COMPILE_FLAG="${IS_CMAKE_COMPILE_FLAG:-1}"

EXEC_SPACE="${EXEC_SPACE:-device}"

CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-90}"

# if [ "$DELETE_LOG" -eq 1 ]; then
#     rm -f sfem_*.out sfem_*.err
# fi

cd "${BUILD_DIR}"

if [ "$IS_CMAKE_COMPILE_FLAG" -eq 0 ]; then
    if [ "$IS_DEBUG" -eq 1 ]; then
        COMPLIE_FLAG="-DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DSFEM_ENABLE_CUDA=ON -DSFEM_ENABLE_PYTHON=OFF -DSFEM_ENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DSFEM_CUDA_ARCH=${CUDA_ARCHITECTURES} -DCMAKE_BUILD_TYPE=Debug  -DCMAKE_CUDA_FLAGS='-G -O0'"
    else
        COMPLIE_FLAG="-DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DSFEM_ENABLE_CUDA=ON -DSFEM_ENABLE_PYTHON=OFF -DSFEM_ENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DSFEM_CUDA_ARCH=${CUDA_ARCHITECTURES} -DCMAKE_BUILD_TYPE=Release"
    fi
    srun -u --uenv=prgenv-gnu/24.7:v3 --view=default \
     bash -lc "cmake .. ${COMPLIE_FLAG}"
fi




# export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
    # export OMP_PLACES=cores
    # export OMP_PROC_BIND=close


# compile flag in daint

srun -u --uenv=prgenv-gnu/24.7:v3 --view=default \
     bash -lc "make -j12"

srun -u --uenv=prgenv-gnu/24.7:v3 --view=default \
     bash -lc "SFEM_EXECUTION_SPACE=${EXEC_SPACE} ./${BIN_NAME} ${BIN_ARGS}"





