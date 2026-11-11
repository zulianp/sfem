#!/bin/bash
#SBATCH --job-name=movie
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/movie_%j.out
#SBATCH --uenv=paraview/6.1:v2
#SBATCH --view=default
# The frames of a stroke run's movie (render_strokes.py): every exported step, 25 frames per process
# (the EGL driver of the Alps nodes aborts a process after 25 frames), JOBS processes at once over
# the node's four GPUs; FRAMES renders only the listed frames, in one process (a check of the views);
# FIRSTS only the 25-frame chunks starting there (to fill those whose process died: pvbatch can stop
# on a bus error at start-up when 16 start at once).
# Into OUT, a new folder (the job stops if it exists). Encode with
# ffmpeg -framerate 30 -i %05d.png -c:v libx264 -pix_fmt yuv420p strokes.mp4.
#   SIM=<run>/sim2 OUT=<folder> [PUMP=LDP24A] [LOAD_FACTOR=0.919119] [FRAMES=k1,k2,...] [FIRSTS="k1 k2 ..."]
#   [JOBS=16]
set -u
ulimit -c 0
SRC=/ritom/scratch/cscs/zulianp/sfem-prony-md
R=$SRC/workflows/ldp_containment/transient/render_strokes.py
SIM=${SIM:?SIM=<run>/sim2}
OUT=${OUT:?OUT=<new folder>}
[ -e "$OUT" ] && { echo "movie: $OUT exists; give a new OUT"; exit 1; }
mkdir -p "$OUT"
opts="--pump=${PUMP:-LDP24A} --load-factor=${LOAD_FACTOR:-0.919119}"
if [ -n "${FRAMES:-}" ]; then
  VTK_DEFAULT_EGL_DEVICE_INDEX=0 pvbatch $R "$SIM" "$OUT" $opts --frames=$FRAMES > "$OUT/render.log" 2>&1
  echo "rc=$?"
else
  n=$(ls "$SIM/out" | grep -c '^disp\.0\.')
  JOBS=${JOBS:-16}
  k=0
  for first in ${FIRSTS:-$(seq 0 25 $((n - 1)))}; do
    VTK_DEFAULT_EGL_DEVICE_INDEX=$((k % 4)) pvbatch $R "$SIM" "$OUT" $opts --first=$first --count=25 \
      > "$OUT/render_$first.log" 2>&1 &
    k=$((k + 1))
    [ $((k % JOBS)) -eq 0 ] && wait
  done
  wait
  echo "steps=$n"
fi
echo "frames=$(ls "$OUT" | grep -c '^[0-9]*\.png$')"
