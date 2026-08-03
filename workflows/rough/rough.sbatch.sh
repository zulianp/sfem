#!/bin/bash
#SBATCH --job-name=rough
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=slurm-rough-%j.out
#SBATCH --error=slurm-rough-%j.err
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --uenv-passthrough=use

set -euo pipefail

export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

export PATH=/capstor/scratch/cscs/zulianp/installations/sfem/bin/:$PATH

echo "#---------------#"
date
echo "#---------------#"

set -x

SFEM_MAX_COARSE_IT=400000       \
SFEM_COARSE_OP_TYPE=MF          \
SFEM_MAX_INNER_IT=4             \
SFEM_ENABLE_LINE_SEARCH=1       \
SFEM_MAX_IT=200                 \
SFEM_ATOL=5e-7                  \
SFEM_ELEMENT_REFINE_LEVEL=4     \
SFEM_STAGNATION_THRESHOLD=10    \
SFEM_PENALTY_PARAM=100          \
SFEM_NL_SMOOTH_STEPS=60         \
SMESH_TRACE_FILE=obs.csv        \
        obs rock ./sdf dirichlet.yaml rock/contact_boundary rough_output

cat obs.csv

source $SCRATCH/sfem/venv/bin/activate

# raw_to_db rough_output/mesh rough_output.vtk -p 'rough_output/out/*.*'

rm -f rough_output/out/contact_stress.{1,2}.*
rm -f rough_output/out/rhs.{0,1,2}.*

SFEM_TRACE_FILE=hex8_cauchy_stress.csv \
	hex8_cauchy_stress rough_output/mesh 1 1 rough_output/out/disp.0.float64 rough_output/out/disp.1.float64 rough_output/out/disp.2.float64 rough_output/out/cauchy_stress

raw_to_db rough_output/mesh rough_output.vtk -p 'rough_output/out/*.float64' $EXTRA_OPTIONS
raw_to_db rough_output/coarse_mesh macro_mesh.vtk


echo "#---------------#"
date
echo "#---------------#"