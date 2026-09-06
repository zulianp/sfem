#!/bin/bash -l
#SBATCH --job-name=mr_replay
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --output=replay-%j.out

set -euo pipefail

# Submit from inside the checkout; Slurm runs a spooled copy of this file.
ROOT_DIR="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$ROOT_DIR"

# Activate the same compiler/MPI and Python environment used for the build
# before submitting, or add the site's environment setup here.
export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export SFEM_BUILD_DIR="${SFEM_BUILD_DIR:-$ROOT_DIR/build_release}"
export SFEM_TEST_TARGET=sfem_MooneyRivlinGravityTest
export SFEM_TEST_OUTPUT_SUBDIR=test_mooney_rivlin_gravity

# Short pilot first; use SFEM_T=7.5 for the full experiment.
export SFEM_BASE_RESOLUTION="${SFEM_BASE_RESOLUTION:-15}"
export SFEM_DT="${SFEM_DT:-0.005}"
export SFEM_T="${SFEM_T:-0.5}"
export SFEM_DENSITY=1
export SFEM_C10=800.622
export SFEM_C01=800.108
export SFEM_BULK_MODULUS=4000
export SFEM_PRONY_G="0.4,0.4,0.1,0.05"
export SFEM_PRONY_TAU="1,2,5,10"
export SFEM_USE_WLF=0
export SFEM_NEWTON_DAMPING=1
export SFEM_ENABLE_CONTACT=0
export SFEM_CONTACT_DIR=0
export SFEM_FIX_SIDE=0
export SFEM_GRAVITY_DIR=1
export SFEM_GRAVITY=0.5

# Suppress periodic field exports; replay CSV still records every step.
export SFEM_EXPORT_FREQ=1000000
OUT_BASE="$SFEM_BUILD_DIR/replay_${SLURM_JOB_ID}"
export SFEM_MESH="$OUT_BASE/mesh"
mkdir -p "$OUT_BASE"

python3 python/sfem/mesh/box_mesh.py "$SFEM_MESH" \
    -c hex8 \
    -x "$((10 * SFEM_BASE_RESOLUTION + 1))" \
    -y "$((SFEM_BASE_RESOLUTION + 1))" \
    -z "$((SFEM_BASE_RESOLUTION + 1))" \
    --width=10 --height=1 --depth=1

run_replay() {
    local storage="$1"
    local scaling="$2"

    srun --cpu-bind=cores bash scripts/run_mr_visco_case.sh \
        --history-mode per_qp \
        --history-storage "$storage" \
        --history-scaling "$scaling" \
        --replay \
        --out "$OUT_BASE/per_qp_${storage}_${scaling}"
}

run_replay float64 none
run_replay float32 none
run_replay float16 none
run_replay float16 tensor
run_replay float16 element_prony
