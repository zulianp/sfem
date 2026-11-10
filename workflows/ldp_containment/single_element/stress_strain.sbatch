#!/bin/bash
#SBATCH --job-name=stress-strain
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/stress_strain_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The one-element stress-strain curves (stress_strain.py) on the host, one core per run, into OUT
# (a new folder; the script stops if it exists). ARGS: further options of stress_strain.py.
#   [OUT=runs/stress_strain] [ARGS="--rates 1e1,1e5,1e7,1e9 --terms 4,5,6,15"]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
OUT=${OUT:-$D/runs/stress_strain}
# shellcheck disable=SC2086
srun -n 1 -c 72 $D/venv/bin/python $D/workflows/ldp_containment/single_element/stress_strain.py \
    $D/build-cuda/playground $OUT ${ARGS:-}
