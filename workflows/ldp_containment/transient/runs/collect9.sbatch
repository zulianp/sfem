#!/bin/bash
#SBATCH --job-name=collect9
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/collect9_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The LDP paper's valve data from the tangent x history campaign (tangent9_valves.sbatch): every
# finished run of each mesh against bsr32__full_a (valve_runs.csv, valve_series.csv), the repeat
# floor (each finished BSR f32 repeat in turn the reference, sets <mesh>_ref_<r>), the valve events
# and the history snapshots against the reference's; a mesh whose reference did not finish is left
# out, and said so. Then, per pump:
#   FF200 (the default; ldp_cases/FF200_ExodusMesh, TAG tp9, strokes to 0.18 s): the blocks' elements,
#     the TET10 mesh's tangent bytes, the transient runs (collect_transient.py) and the torsion rows of
#     runs/paper_data8; into runs/paper_data_<TAG> (DATA).
#   LDP24A (ldp_cases/LDP24A_r100, TAG la, strokes to 0.1655 s): the tangent's bytes from both
#     meshes' node graphs, the blocks' elements of the campaign's mesh and of the refined one, and the
#     strokes' references (collect_transient.py, <out>/strokes); into runs/paper_la (OUT).
# TAG names the campaign's folders (tangent9_valves.sbatch's PREFIX without _s0/_p2). The output
# folder is a new one: the job stops if it exists.
#   [PUMP=FF200|LDP24A] [TAG=<campaign>] [DATA=<folder under runs> | OUT=<path>] [PAPER_PY=xsesfsi_papers/ldp_containment/python]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
PY=$D/venv/bin/python
# The paper's collectors (the paper lives in the private repository xsesfsi_papers, ldp_containment).
PAPER=${PAPER_PY:-/ritom/scratch/cscs/zulianp/xsesfsi_papers/ldp_containment/python}
COL=$PAPER/collect_prony.py
PUMP=${PUMP:-FF200}
case $PUMP in
  FF200)  CASE=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh; TAG=${TAG:-tp9}; T_END=0.18
          P=${OUT:-$D/runs/${DATA:-paper_data_$TAG}} ;;
  LDP24A) CASE=/ritom/scratch/cscs/zulianp/ldp_cases/LDP24A_r100; TAG=${TAG:-la}; T_END=0.1655
          P=${OUT:-$D/runs/${DATA:-paper_la}} ;;
  *) echo "collect9: unknown PUMP $PUMP"; exit 1 ;;
esac
[ -e $P ] && { echo "collect9: $P exists; give a new DATA or OUT"; exit 1; }
mkdir -p $P
for spec in TET4:${TAG}_s0_ TET10:${TAG}_p2_; do
  el=${spec%:*}; prefix=${spec#*:}
  # The finished runs: the history reaches the end of the strokes; the smoke runs (_s0) are not.
  runs=""
  for d in $CASE/${prefix}*; do
    r=${d#$CASE/$prefix}
    case $r in *_s0) continue ;; esac
    h=$d/sim2/history.csv
    if [ -f $h ] && tail -1 $h | awk -F, -v t=$T_END '{exit !($2 > t - 1e-9)}'; then runs="$runs $r"; else echo "$el $r unfinished"; fi
  done
  # Every error is measured against the reference: without it finished, the mesh is left out.
  case " $runs " in *" bsr32__full_a "*) ;; *) echo "$el: the reference bsr32__full_a did not finish; $el left out"; continue ;; esac
  # shellcheck disable=SC2086
  $PY $COL runs $P/valve_runs.csv $P/valve_series.csv $el $CASE bsr32__full_a $runs --prefix $prefix > $P/runs_$el.log 2>&1
  echo "runs $el rc=$? ($(echo $runs | wc -w) runs)"
  # The repeat floor over the finished repeats only, each in turn the reference.
  for ref in b c d; do
    case " $runs " in *" bsr32__full_$ref "*) ;; *) echo "$el floor: bsr32__full_$ref unfinished or missing"; continue ;; esac
    others=""
    for x in a b c d; do [[ $x > $ref ]] && case " $runs " in *" bsr32__full_$x "*) others="$others bsr32__full_$x" ;; esac; done
    [ -z "$others" ] && continue
    # shellcheck disable=SC2086
    $PY $COL runs $P/valve_runs.csv $P/valve_series_floor.csv ${el}_ref_$ref $CASE bsr32__full_$ref $others \
        --prefix $prefix >> $P/runs_$el.log 2>&1
    echo "floor $el ref $ref rc=$? ($others)"
  done
  # shellcheck disable=SC2086
  $PY $COL events $P/valve_events.csv $el $CASE $runs --prefix $prefix >> $P/runs_$el.log 2>&1
  echo "events $el rc=$?"
  snaps=""; for r in $runs; do [ -d $CASE/$prefix$r/sim2/history ] && [ $r != bsr32__full_a ] && snaps="$snaps $r"; done
  # shellcheck disable=SC2086
  [ -n "$snaps" ] && $PY $COL snapshot $P/history_snapshots.csv $el $CASE bsr32__full_a $snaps --prefix $prefix >> $P/runs_$el.log 2>&1
  echo "snapshots $el rc=$? ($snaps)"
done
if [ $PUMP = FF200 ]; then
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
else
  $PY $COL tangent $P/prony_tangent_LDP24A.csv $CASE/${TAG}_s0_bsr32__full_a/sim2/mesh
  $PY $COL tangent $P/prony_tangent_LDP24A_p2.csv $CASE/${TAG}_p2_bsr32__full_a/sim2/mesh
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
fi
$PY - <<PYEOF
import csv
for r in csv.DictReader(open("$P/valve_runs.csv")):
    t = sum(float(r["bytes_%s_device" % c] or 0) for c in ("tangent_values", "tangent_structure", "tangent_work"))
    # bytes_stress (the relaxed-stress slots, runs before sfem#103) or bytes_previous (after): one is empty.
    s = sum(float(r.get(k) or 0) for k in ("bytes_history", "bytes_reference", "bytes_frozen", "bytes_stress", "bytes_previous"))
    print("%-10s %-36s e_u med %-9.3g max %-9.3g shift %-6s failed %-3s t %-8s tangent %8.2f MB prony %8.2f MB peak %8.1f MB" % (
        r["set"], r["run"], float(r["e_u_median"]), float(r["e_u_max"]), r["event_shift_ms"], r["failed"], r["stepper_s"],
        t / 1e6, s / 1e6, float(r["device_used_peak"]) / 1e6))
PYEOF
