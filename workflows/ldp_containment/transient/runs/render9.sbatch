#!/bin/bash
#SBATCH --job-name=render9
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/render9_%j.out
#SBATCH --uenv=paraview/6.1:v2
#SBATCH --view=default
# The LDP paper's history renders: the per-element error of the stored terms' overstress (Hsum, from
# collect9.sbatch's snapshot pass) on the TET10 valves' surfaces, for every TET10 run with
# history snapshots, at the snapshot times of tangent9_valves.sbatch, one colour range for all (log10
# -8 to -1: a repeat in double to float16 averages). int8, float16 scaled and the quantized BSR i8
# are out of the paper, and so are the smoke runs.
#   [PUMP=FF200|LDP24A] [TAG=<campaign>] [TIMES=t0,t1,...] [RENDERS=<folder under runs>]
# PUMP: FF200 (the default; ldp_cases/FF200_ExodusMesh, TAG tp9, snapshot times 0.012,0.0255,0.043,
# RENDERS history9) or LDP24A (ldp_cases/LDP24A_r100, TAG la, RENDERS history_la; TIMES is required:
# the SNAP its runs were given, which the runs do not record). TIMES must name as many times as the
# reference has snapshots. RENDERS is a new folder: the job stops if it exists.
set -u
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
SRC=$SFEM_ROOT
case ${PUMP:-FF200} in
  FF200)  CASE=$LDP_CASES/FF200_ExodusMesh; TAG=${TAG:-tp9}; TIMES=${TIMES:-0.012,0.0255,0.043}; RENDERS=${RENDERS:-history9} ;;
  LDP24A) CASE=$LDP_CASES/LDP24A_r100; TAG=${TAG:-la}; RENDERS=${RENDERS:-history_la}
          [ -n "${TIMES:-}" ] || { echo "render9: LDP24A needs TIMES, the snapshot times (SNAP) its runs were given"; exit 1; } ;;
  *) echo "render9: unknown PUMP ${PUMP}"; exit 1 ;;
esac
O=$SFEM_RUNS/$RENDERS
[ -e $O ] && { echo "render9: $O exists; give a new RENDERS"; exit 1; }
# One time a snapshot: checked against the reference's before anything is written.
ref=$CASE/${TAG}_p2_bsr32__full_a/sim2/history
n_snap=$(find $ref -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
n_times=$(echo "$TIMES" | tr ',' '\n' | grep -c .)
[ "$n_snap" -eq "$n_times" ] || { echo "render9: $ref has $n_snap snapshots, TIMES names $n_times"; exit 1; }
runs=""
for d in $CASE/${TAG}_p2_*; do
  r=${d#$CASE/${TAG}_p2_}
  case $r in *_s0 | qbsr8__* | *int8_* | *float16_scaled*) continue ;; esac
  [ $r != bsr32__full_a ] && [ -f $d/sim2/history/0/PRESSURE_VALVE.iso.Hsum.err.float64 ] && runs="$runs,$r"
done
VTK_DEFAULT_EGL_DEVICE_INDEX=0 $SFEM_PVBATCH $SRC/workflows/ldp_containment/transient/render_history.py $CASE $O \
  --prefix=${TAG}_p2_ --reference=bsr32__full_a --runs=${runs#,} --times=$TIMES --field=Hsum --range=-8,-1 \
  --size=1200,900 --pump=${PUMP:-FF200}
ls $O | wc -l
