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

source $SCRATCH/sfem/venv/bin/activate
export PATH=/capstor/scratch/cscs/zulianp/installations/sfem/bin/:$PATH

echo "#---------------#"
date
echo "#---------------#"

srun run.sh

cat obs.csv

echo "#---------------#"
date
echo "#---------------#"