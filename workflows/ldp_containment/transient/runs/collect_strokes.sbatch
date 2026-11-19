#!/bin/bash
#SBATCH --job-name=collect-strokes
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/collect_strokes_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The strokes' data files of the LDP paper from the campaign's references (tp10_{s0,p2}_bsr32__full_a,
# python/collect_transient.py) into OUT, a new folder (the collector stops if it exists).
#   [OUT=runs/paper_strokes10]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
OUT=${OUT:-$D/runs/paper_strokes10}
$D/venv/bin/python ${PAPER_PY:-/ritom/scratch/cscs/zulianp/xsesfsi_papers/ldp_containment/python}/collect_transient.py $OUT || exit 1
wc -l $OUT/*.csv
