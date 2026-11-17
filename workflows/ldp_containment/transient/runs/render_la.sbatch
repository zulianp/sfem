#!/bin/bash
#SBATCH --job-name=render-la
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/render_la_%j.out
#SBATCH --uenv=paraview/6.1:v2
#SBATCH --view=default
# The LDP paper's history renders on LDP24A: the per-element error of the stored terms' overstress
# (Hsum, from collect_la.sbatch's snapshot pass) on the TET10 valves' surfaces, for every TET10 run
# with history snapshots (la_p2), at the snapshot times, one colour range for all (log10 -8 to -1);
# into runs/<RENDERS>, a new folder (the job stops if it exists).
#   [RENDERS=history_la]
set -u
SRC=/ritom/scratch/cscs/zulianp/sfem-prony-md
CASE=/ritom/scratch/cscs/zulianp/ldp_cases/LDP24A_r100
O=$SRC/runs/${RENDERS:-history_la}
[ -e $O ] && { echo "render_la: $O exists; give a new RENDERS"; exit 1; }
runs=""
for d in $CASE/la_p2_*; do
  r=${d#$CASE/la_p2_}
  [ $r != bsr32__full_a ] && [ -f $d/sim2/history/0/PRESSURE_VALVE.iso.Hsum.err.float64 ] && runs="$runs,$r"
done
VTK_DEFAULT_EGL_DEVICE_INDEX=0 pvbatch $SRC/workflows/ldp_containment/transient/render_history.py $CASE $O \
  --prefix=la_p2_ --reference=bsr32__full_a --runs=${runs#,} --times=0.012,0.0255,0.043 --field=Hsum --range=-8,-1 \
  --size=1200,900 --pump=LDP24A
ls $O | wc -l
