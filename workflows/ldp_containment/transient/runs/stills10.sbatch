#!/bin/bash
#SBATCH --job-name=ldp-stills10
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/stills10_%j.out
#SBATCH --uenv=paraview/6.1:v2
#SBATCH --view=default
# The paper's stills from a TET4 campaign reference: each view on its own at 1200 x 900; the run cuts
# no step back, so frame n is at n x 0.1 ms. Into OUT, a new folder (the job stops if it exists).
#   [PUMP=FF200|LDP24A] [FRAMES=k1,k2,...] [OUT=runs/<folder>]
# PUMP: FF200 (the default; tp10_s0_bsr32__full_a, frames 0 (at rest), 38 (3.8 ms, the pressure valve
# on its cage), 260 (26 ms, both seated) and 275 (27.5 ms, the suction valve on its stop), OUT
# runs/stills10) or LDP24A (la_s0_bsr32__full_a at the load factor 0.919119, OUT runs/stills_la;
# FRAMES is required).
set -u
SRC=/ritom/scratch/cscs/zulianp/sfem-prony-md
R=$SRC/workflows/ldp_containment/transient/render_strokes.py
case ${PUMP:-FF200} in
  FF200)  SIM=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh/tp10_s0_bsr32__full_a/sim2; FRAMES=${FRAMES:-0,38,260,275}
          OUT=${OUT:-$SRC/runs/stills10}; ARGS="" ;;
  LDP24A) SIM=/ritom/scratch/cscs/zulianp/ldp_cases/LDP24A_r100/la_s0_bsr32__full_a/sim2; OUT=${OUT:-$SRC/runs/stills_la}
          ARGS="--pump=LDP24A --load-factor=0.919119"
          [ -n "${FRAMES:-}" ] || { echo "stills10: LDP24A needs FRAMES"; exit 1; } ;;
  *) echo "stills10: unknown PUMP ${PUMP}"; exit 1 ;;
esac
[ -e $OUT ] && { echo "stills10: $OUT exists; give a new OUT"; exit 1; }
mkdir -p $OUT
# shellcheck disable=SC2086
cd $OUT && VTK_DEFAULT_EGL_DEVICE_INDEX=0 pvbatch $R $SIM $OUT $ARGS --frames=$FRAMES --panels --size=1200,900 > $OUT/stills.log 2>&1
echo "rc=$? $(ls $OUT/*.png | wc -l) files"
