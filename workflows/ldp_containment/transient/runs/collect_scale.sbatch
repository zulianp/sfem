#!/bin/bash
#SBATCH --job-name=collect-scale
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/collect_scale_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The storage at about ten times the size: the pump's mesh refined once (db/solid_r1: FF200's
# scale_mesh.sbatch, LDP24A's prepare.sbatch), TET4 and its TET10 promotion, ten steps from rest per
# configuration of the memory figure (tangent9_valves.sbatch, NO_IC=1, PREFIX <TAG>_s0 and <TAG>_p2).
# Every run's measured bytes into OUT (a new folder; the script stops if it exists), sets TET4x8 and
# TET10x8, against bsr32__full_a. TAG4/TAG10 and T_END4/T_END10 take each mesh's runs from their own
# folders and span. The defaults are FF200's campaign (sc10); LDP24A's (collect_la_studies.sbatch) is
# CASE=ldp_cases/LDP24A_r100 TAG=sc TAG10=sc1 T_END10=1e-4: its TET10x8 runs take one step, the
# storage being the same after one step as after ten, as TET4x8's runs show.
#   [CASE=ldp_cases/FF200_ExodusMesh] [TAG=sc10] [TAG4=TAG] [TAG10=TAG] [T_END4=1e-3] [T_END10=1e-3]
#   [OUT=runs/paper_scale for sc10, runs/paper_scale_<TAG> otherwise]
set -u
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
PY=$SFEM_PY
COL=$PAPER_PY/collect_prony.py
CASE=${CASE:-$LDP_CASES/FF200_ExodusMesh}
TAG=${TAG:-sc10}
TAG4=${TAG4:-$TAG}; TAG10=${TAG10:-$TAG}
if [ $TAG = sc10 ]; then P=${OUT:-$SFEM_RUNS/paper_scale}; else P=${OUT:-$SFEM_RUNS/paper_scale_$TAG}; fi
[ -e $P ] && { echo "collect_scale: $P exists; give a new OUT"; exit 1; }
mkdir -p $P
for spec in TET4x8:${TAG4}_s0_:${T_END4:-1e-3} TET10x8:${TAG10}_p2_:${T_END10:-1e-3}; do
  set_name=${spec%%:*}; rest=${spec#*:}; prefix=${rest%:*}; t_end=${rest#*:}
  runs=""
  for d in $CASE/${prefix}*; do
    r=${d#$CASE/$prefix}
    [ $r = bsr32__full_a ] && continue
    h=$d/sim2/history.csv
    if [ -f $h ] && tail -1 $h | awk -F, -v t=$t_end '{exit !($2 > t - 1e-9)}'; then runs="$runs $r"; else echo "$set_name $r unfinished"; fi
  done
  # shellcheck disable=SC2086
  $PY $COL runs $P/scale_runs.csv $P/scale_series.csv $set_name $CASE bsr32__full_a bsr32__full_a $runs --prefix $prefix \
      > $P/runs_$set_name.log 2>&1
  echo "runs $set_name rc=$? ($(echo $runs | wc -w) runs)"
done
$PY - $P/scale_runs.csv <<'PYEOF'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    # The relaxed-stress slots (bytes_stress) of runs before sfem#103, the previous displacement
    # (bytes_previous) after it: each run fills one, the other is empty.
    state = sum(float(r.get(k) or 0) for k in ("bytes_history", "bytes_reference", "bytes_frozen", "bytes_stress", "bytes_previous")) / 1e6
    tangent = sum(float(r["bytes_%s_device" % c] or 0) for c in ("tangent_values", "tangent_structure", "tangent_work")) / 1e6
    print("%-8s %-32s dofs %-9s elements %-8s steps %-3s failed %-2s tangent %8.1f MB prony %8.1f MB peak %8.0f MB" % (
        r["set"], r["run"], r["dofs"], r["elements"], r["steps"], r["failed"], tangent, state, float(r["device_used_peak"]) / 1e6))
PYEOF
