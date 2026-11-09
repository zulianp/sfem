#!/bin/bash
#SBATCH --job-name=strip-render
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/strip_render_%j.out
#SBATCH --uenv=paraview/6.1:v2
#SBATCH --view=default
# The strip's renders (render_strip.py): the paper's sequence at the times TIMES, without text, and
# the movie's frames, every STRIDE-th exported step with the time stamped, 20 per pvbatch process (the
# EGL driver aborts one after about 25 images); one colour range for both, RANGE (m).
#   RUN=<run folder> OUT=<folder> RANGE=0,<umax> [TIMES=t0,t1,...] [STRIDE=10] [SIZE=1600,700]
set -u
T=/ritom/scratch/cscs/zulianp/sfem-prony-md/workflows/prony_torsion/strip/render_strip.py
export VTK_DEFAULT_EGL_DEVICE_INDEX=0
mkdir -p $OUT
if [ -n "${TIMES:-}" ]; then
  pvbatch $T $RUN $OUT/sequence --times=$TIMES --range=$RANGE --size=${SIZE:-2400,1300} --prefix=seq --crop
fi
n=$(wc -l < $RUN/results/out/time.txt)
steps=$(seq 0 ${STRIDE:-10} $((n - 1)))
echo "$steps" | xargs -n 20 | while read chunk; do
  pvbatch $T $RUN $OUT/movie --steps=$(echo $chunk | tr ' ' ',') --range=$RANGE --size=${SIZE:-1600,700} --stamp --prefix=frame \
    || echo "render_strip: chunk starting $(echo $chunk | cut -d' ' -f1) failed"
done
ls $OUT/sequence $OUT/movie 2>/dev/null | wc -l
