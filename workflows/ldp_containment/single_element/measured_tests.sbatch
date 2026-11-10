#!/bin/bash
#SBATCH --job-name=measured-tests
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/measured_tests_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The tests of 70 EPDM 281 on one element (measured_tests.py) on the host, one core per run, into
# OUT (a new folder; the script stops if it exists). DATA: the extracted measurements.
#   DATA=<folder> [OUT=runs/measured_tests] [ARGS="--terms 4,15 --steps 2000"]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
OUT=${OUT:-$D/runs/measured_tests}
# shellcheck disable=SC2086
srun -n 1 -c 72 $D/venv/bin/python $D/workflows/ldp_containment/single_element/measured_tests.py \
    $D/build-cuda/playground $DATA $OUT ${ARGS:-}
