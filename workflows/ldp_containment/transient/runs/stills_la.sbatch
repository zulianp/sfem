#!/bin/bash
#SBATCH --job-name=ldp-stills-la
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/stills_la_%j.out
#SBATCH --uenv=paraview/6.1:v2
#SBATCH --view=default
# The paper's stills from LDP24A's TET4 campaign reference (la_s0_bsr32__full_a): each view on its own
# at 1200 x 900, the frames FRAMES (the run cuts no step back, so frame n is at n x 0.1 ms). Into OUT,
# a new folder (the job stops if it exists).
#   FRAMES=k1,k2,... [OUT=runs/stills_la]
set -u
SRC=/ritom/scratch/cscs/zulianp/sfem-prony-md
R=$SRC/workflows/ldp_containment/transient/render_strokes.py
SIM=/ritom/scratch/cscs/zulianp/ldp_cases/LDP24A_r100/la_s0_bsr32__full_a/sim2
OUT=${OUT:-$SRC/runs/stills_la}
[ -e $OUT ] && { echo "stills_la: $OUT exists; give a new OUT"; exit 1; }
mkdir -p $OUT
cd $OUT && VTK_DEFAULT_EGL_DEVICE_INDEX=0 pvbatch $R $SIM $OUT --pump=LDP24A --load-factor=0.919119 --frames=$FRAMES \
    --panels --size=1200,900 > $OUT/stills.log 2>&1
echo "rc=$? $(ls $OUT/*.png | wc -l) files"
