#!/bin/bash
#SBATCH --job-name=collect9
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/collect9_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The LDP paper's valve data from the tangent x history campaign (tangent9_valves.sbatch), into
# runs/paper_data9: every finished run of each mesh against bsr32__full_a (valve_runs.csv,
# valve_series.csv), the repeat floor (each BSR f32 repeat in turn the reference, sets <mesh>_ref_<r>),
# the valve events, the history snapshots against the reference's, the TET10 mesh's analytic tangent
# bytes, the transient runs (collect_transient.py), and the torsion rows of runs/paper_data8.
# TAG names the campaign's folders (tangent9_valves.sbatch's PREFIX without _s0/_p2; default tp9),
# DATA the output folder under runs (default paper_data_<TAG>), a new one: the script stops if it exists.
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
PY=$D/venv/bin/python
PAPER=$D/dev/wip/ldp_containment/python
COL=$PAPER/collect_prony.py
CASE=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh
TAG=${TAG:-tp9}
P=$D/runs/${DATA:-paper_data_$TAG}
[ -e $P ] && { echo "collect9: $P exists; give a new DATA"; exit 1; }
mkdir -p $P
for spec in TET4:${TAG}_s0_ TET10:${TAG}_p2_; do
  el=${spec%:*}; prefix=${spec#*:}
  # The finished runs: the history reaches the end of the strokes; the smoke runs (_s0) are not.
  runs=""
  for d in $CASE/${prefix}*; do
    r=${d#$CASE/$prefix}
    case $r in *_s0) continue ;; esac
    h=$d/sim2/history.csv
    if [ -f $h ] && tail -1 $h | awk -F, '{exit !($2 > 0.18 - 1e-9)}'; then runs="$runs $r"; else echo "$el $r unfinished"; fi
  done
  $PY $COL runs $P/valve_runs.csv $P/valve_series.csv $el $CASE bsr32__full_a $runs --prefix $prefix > $P/runs_$el.log 2>&1
  echo "runs $el rc=$? ($(echo $runs | wc -w) runs)"
  for ref in b c d; do
    others=""; for x in a b c d; do [[ $x > $ref ]] && others="$others bsr32__full_$x"; done
    [ -n "$others" ] && $PY $COL runs $P/valve_runs.csv $P/valve_series_floor.csv ${el}_ref_$ref $CASE bsr32__full_$ref $others \
        --prefix $prefix >> $P/runs_$el.log 2>&1
  done
  $PY $COL events $P/valve_events.csv $el $CASE $runs --prefix $prefix >> $P/runs_$el.log 2>&1
  echo "events $el rc=$?"
  snaps=""; for r in $runs; do [ -d $CASE/$prefix$r/sim2/history ] && [ $r != bsr32__full_a ] && snaps="$snaps $r"; done
  [ -n "$snaps" ] && $PY $COL snapshot $P/history_snapshots.csv $el $CASE bsr32__full_a $snaps --prefix $prefix >> $P/runs_$el.log 2>&1
  echo "snapshots $el rc=$? ($snaps)"
done
# The element blocks of each mesh (each keeps its own history store, so patches do not span them).
echo "set,block,elements" > $P/valve_blocks.csv
for spec in TET4:${TAG}_s0_ TET10:${TAG}_p2_; do
  for d in $CASE/${spec#*:}bsr32__full_a/sim2/mesh/blocks/*/; do
    echo "${spec%:*},$(basename $d),$(( $(stat -c %s $d/i0.int32) / 4 ))" >> $P/valve_blocks.csv
  done
done
$PY $COL tangent $P/prony_tangent_FF200_p2.csv $CASE/${TAG}_p2_bsr32__full_a/sim2/mesh
$PY $COL tangent $P/prony_tangent_FF200.csv $CASE/${TAG}_s0_bsr32__full_a/sim2/mesh
python3 $PAPER/collect_transient.py $P > $P/transient.log 2>&1; echo "transient rc=$?"
# The torsion bar's rows, collected by collect8.sbatch.
for f in runs series; do
  $PY - $D/runs/paper_data8/prony_$f.csv $P/torsion_$f.csv <<'PYEOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["set"].startswith("torsion")]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
PYEOF
done
wc -l $P/*.csv
$PY - <<PYEOF
import csv
for r in csv.DictReader(open("$P/valve_runs.csv")):
    t = sum(float(r["bytes_%s_device" % c] or 0) for c in ("tangent_values", "tangent_structure", "tangent_work"))
    s = sum(float(r[k] or 0) for k in ("bytes_history", "bytes_reference", "bytes_frozen", "bytes_stress"))
    print("%-10s %-36s e_u med %-9.3g max %-9.3g shift %-6s failed %-3s t %-8s tangent %8.2f MB prony %8.2f MB peak %8.1f MB" % (
        r["set"], r["run"], float(r["e_u_median"]), float(r["e_u_max"]), r["event_shift_ms"], r["failed"], r["stepper_s"],
        t / 1e6, s / 1e6, float(r["device_used_peak"]) / 1e6))
PYEOF
