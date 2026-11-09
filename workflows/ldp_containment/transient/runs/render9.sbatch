#!/bin/bash
#SBATCH --job-name=render9
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/render9_%j.out
#SBATCH --uenv=paraview/6.1:v2
#SBATCH --view=default
# The LDP paper's history renders: the per-element error of the stored terms' overstress (Hsum, from
# collect9.sbatch's snapshot pass) on the TET10 valves' surfaces, for every TET10 run with
# history snapshots, at the snapshot times of tangent9_valves.sbatch, one colour range for all (log10
# -8 to -1: a repeat in double to float16 averages); into
# runs/history9. TAG names the campaign's folders (default tp9), RENDERS the output folder under runs
# (a new one: the job stops if it exists). int8 a.+d. and float16 scaled are out of the paper.
set -u
SRC=/ritom/scratch/cscs/zulianp/sfem-prony-md
CASE=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh
TAG=${TAG:-tp9}
O=$SRC/runs/${RENDERS:-history9}
[ -e $O ] && { echo "render9: $O exists; give a new RENDERS"; exit 1; }
runs=""
for d in $CASE/${TAG}_p2_*; do
  r=${d#$CASE/${TAG}_p2_}
  case $r in *_s0 | *int8_md* | *float16_scaled*) continue ;; esac
  [ $r != bsr32__full_a ] && [ -f $d/sim2/history/0/PRESSURE_VALVE.iso.Hsum.err.float64 ] && runs="$runs,$r"
done
VTK_DEFAULT_EGL_DEVICE_INDEX=0 pvbatch $SRC/workflows/ldp_containment/transient/render_history.py $CASE $O \
  --prefix=${TAG}_p2_ --reference=bsr32__full_a --runs=${runs#,} --times=0.012,0.0255,0.043 --field=Hsum --range=-8,-1 --size=1200,900
ls $O | wc -l
