#!/bin/bash
#SBATCH --job-name=fit-series
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/fit_series_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The fit of the valves' material to 70 EPDM 281 (fit_series.py) on the host, into OUT (a new folder;
# the script stops if it exists). DATA: the extracted measurements.
#   DATA=<folder> [OUT=runs/fit_series] [ARGS="--slow 4 --steps 1000 --dma-temp 30 --dma-weight 1"]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
OUT=${OUT:-$D/runs/fit_series}
# shellcheck disable=SC2086
srun -n 1 -c 72 $D/venv/bin/python $D/workflows/ldp_containment/single_element/fit_series.py \
    $D/build-cuda/playground $DATA $OUT ${ARGS:-}
