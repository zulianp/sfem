#!/bin/bash
#SBATCH --job-name=ldp-stills10
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/stills10_%j.out
#SBATCH --uenv=paraview/6.1:v2
#SBATCH --view=default
# The paper's stills from the TET4 campaign reference (tp10_s0_bsr32__full_a): each view on its own at
# 1200 x 900, frames 0 (at rest), 38 (3.8 ms, the pressure valve on its cage), 260 (26 ms, both
# seated) and 275 (27.5 ms, the suction valve on its stop); the run cuts no step back, so frame n is
# at n x 0.1 ms. Into OUT, a new folder (the job stops if it exists).
#   [OUT=runs/stills10]
set -u
SRC=/ritom/scratch/cscs/zulianp/sfem-prony-md
R=$SRC/workflows/ldp_containment/transient/render_strokes.py
SIM=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh/tp10_s0_bsr32__full_a/sim2
OUT=${OUT:-$SRC/runs/stills10}
[ -e $OUT ] && { echo "stills10: $OUT exists; give a new OUT"; exit 1; }
mkdir -p $OUT
cd $OUT && VTK_DEFAULT_EGL_DEVICE_INDEX=0 pvbatch $R $SIM $OUT --frames=0,38,260,275 --panels --size=1200,900 > $OUT/stills.log 2>&1
echo "rc=$? $(ls $OUT/*.png | wc -l) files"
