#!/bin/bash -l
#SBATCH --job-name=torsion
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --output=torsion-%j.out

set -euo pipefail

# Submit anywhere inside this checkout. Load the build's uenv before submission
# and pass --uenv-passthrough=use, or specify the matching --uenv explicitly.
ROOT_DIR="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$ROOT_DIR"
if [[ "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo "[error] Torsion supports one MPI rank; use --ntasks=1." >&2
    exit 1
fi
source "${TORSION_VENV:-$ROOT_DIR/.venv}/bin/activate"
export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$(mktemp -d "${TMPDIR:-/tmp}/torsion-mpl-${SLURM_JOB_ID}.XXXXXX")"

srun --ntasks=1 --cpu-bind=cores bash scripts/run_torsion_study.sh
