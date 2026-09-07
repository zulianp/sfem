#!/bin/bash -l
#SBATCH --job-name=mr_coupled
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --output=coupled-%j.out

set -euo pipefail

# Submit from inside the repository, with the build's compiler/MPI uenv loaded.
ROOT_DIR="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$ROOT_DIR"
source "$ROOT_DIR/.venv/bin/activate"

export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export MPLBACKEND=Agg
export SFEM_BUILD_DIR="${SFEM_BUILD_DIR:-$ROOT_DIR/build_release}"
export SFEM_OUT_BASE="$SFEM_BUILD_DIR/coupled_${SLURM_JOB_ID}"
export SFEM_SKIP_BUILD=1
# Reset the replay export setting; retain every 5th coupled displacement step.
export SFEM_EXPORT_FREQ=5

srun --cpu-bind=cores bash scripts/run_mr_visco_history_compare.sh
