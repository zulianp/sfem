#!/bin/bash
#SBATCH --job-name=collect-la
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/collect_la_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The LDP paper's valve data from LDP24A_r100's tangent x history campaign (tangent9_valves.sbatch,
# PREFIX la_s0, la_p2) into OUT, a new folder (the job stops if it exists): every finished run of each
# mesh against bsr32__full_a (valve_runs.csv, valve_series.csv), the repeat floor (each BSR f32
# repeat in turn the reference, sets <mesh>_ref_<r>), the valve events, the history snapshots
# against the reference's, the tangent's bytes from the meshes' node graphs, the blocks' elements
# of the campaign's and the refined mesh, and the strokes' references (collect_transient.py, OUT/strokes).
#   [OUT=runs/paper_la] [PAPER_PY=xsesfsi_papers/ldp_containment/python]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
PY=$D/venv/bin/python
PAPER=${PAPER_PY:-/ritom/scratch/cscs/zulianp/xsesfsi_papers/ldp_containment/python}
COL=$PAPER/collect_prony.py
CASE=/ritom/scratch/cscs/zulianp/ldp_cases/LDP24A_r100
T_END=0.1655
P=${OUT:-$D/runs/paper_la}
[ -e $P ] && { echo "collect_la: $P exists; give a new OUT"; exit 1; }
mkdir -p $P
for spec in TET4:la_s0_ TET10:la_p2_; do
  el=${spec%:*}; prefix=${spec#*:}
  # The finished runs: the history reaches the end of the strokes.
  runs=""
  for d in $CASE/${prefix}*; do
    r=${d#$CASE/$prefix}
    h=$d/sim2/history.csv
    if [ -f $h ] && tail -1 $h | awk -F, -v t=$T_END '{exit !($2 > t - 1e-9)}'; then runs="$runs $r"; else echo "$el $r unfinished"; fi
  done
  # shellcheck disable=SC2086
  $PY $COL runs $P/valve_runs.csv $P/valve_series.csv $el $CASE bsr32__full_a $runs --prefix $prefix > $P/runs_$el.log 2>&1
  echo "runs $el rc=$? ($(echo $runs | wc -w) runs)"
  for ref in b c d; do
    others=""; for x in a b c d; do [[ $x > $ref ]] && others="$others bsr32__full_$x"; done
    # shellcheck disable=SC2086
    [ -n "$others" ] && $PY $COL runs $P/valve_runs.csv $P/valve_series_floor.csv ${el}_ref_$ref $CASE bsr32__full_$ref $others \
        --prefix $prefix >> $P/runs_$el.log 2>&1
  done
  # shellcheck disable=SC2086
  $PY $COL events $P/valve_events.csv $el $CASE $runs --prefix $prefix >> $P/runs_$el.log 2>&1
  echo "events $el rc=$?"
  snaps=""; for r in $runs; do [ -d $CASE/$prefix$r/sim2/history ] && [ $r != bsr32__full_a ] && snaps="$snaps $r"; done
  # shellcheck disable=SC2086
  [ -n "$snaps" ] && $PY $COL snapshot $P/history_snapshots.csv $el $CASE bsr32__full_a $snaps --prefix $prefix >> $P/runs_$el.log 2>&1
  echo "snapshots $el rc=$? ($snaps)"
done
$PY $COL tangent $P/prony_tangent_LDP24A.csv $CASE/la_s0_bsr32__full_a/sim2/mesh
$PY $COL tangent $P/prony_tangent_LDP24A_p2.csv $CASE/la_p2_bsr32__full_a/sim2/mesh
# The blocks' elements: the campaign's mesh (TET10 promotes it) and the refined one.
{ echo "set,block,elements"
  for spec in TET4:solid TET10:solid_p2; do
    for b in $CASE/db/${spec#*:}/blocks/*/; do echo "${spec%:*},$(basename $b),$(( $(stat -c %s $b/i0.int32) / 4 ))"; done
  done; } > $P/valve_blocks.csv
{ echo "set,block,elements"
  for s in TET4x8 TET10x8; do
    for b in $CASE/db/solid_r1/blocks/*/; do echo "$s,$(basename $b),$(( $(stat -c %s $b/i0.int32) / 4 ))"; done
  done; } > $P/scale_blocks.csv
$PY $PAPER/collect_transient.py $P/strokes > $P/transient.log 2>&1; echo "transient rc=$?"
wc -l $P/*.csv $P/strokes/*.csv
$PY - <<PYEOF
import csv
for r in csv.DictReader(open("$P/valve_runs.csv")):
    t = sum(float(r["bytes_%s_device" % c] or 0) for c in ("tangent_values", "tangent_structure", "tangent_work"))
    s = sum(float(r[k] or 0) for k in ("bytes_history", "bytes_reference", "bytes_frozen", "bytes_stress"))
    print("%-10s %-36s e_u med %-9.3g max %-9.3g shift %-6s failed %-3s t %-8s tangent %8.2f MB prony %8.2f MB peak %8.1f MB" % (
        r["set"], r["run"], float(r["e_u_median"]), float(r["e_u_max"]), r["event_shift_ms"], r["failed"], r["stepper_s"],
        t / 1e6, s / 1e6, float(r["device_used_peak"]) / 1e6))
PYEOF
