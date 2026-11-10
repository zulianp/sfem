#!/bin/bash
#SBATCH --job-name=collect-scale
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/collect_scale_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The storage at about ten times the size: FF200 refined once (scale_mesh.sbatch, db/solid_r1), TET4
# and its TET10 promotion, ten steps from rest per configuration of the memory figure
# (tangent9_valves.sbatch, NO_IC=1, PREFIX sc10_s0 and sc10_p2). Every run's measured bytes into
# OUT (a new folder; the script stops if it exists), sets TET4x8 and TET10x8, against bsr32__full_a.
#   [OUT=runs/paper_scale]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
PY=$D/venv/bin/python
COL=${PAPER_PY:-/ritom/scratch/cscs/zulianp/xsesfsi_papers/ldp_containment/python}/collect_prony.py
CASE=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh
P=${OUT:-$D/runs/paper_scale}
[ -e $P ] && { echo "collect_scale: $P exists; give a new OUT"; exit 1; }
mkdir -p $P
for spec in TET4x8:sc10_s0_ TET10x8:sc10_p2_; do
  set_name=${spec%:*}; prefix=${spec#*:}
  runs=""
  for d in $CASE/${prefix}*; do
    r=${d#$CASE/$prefix}
    [ $r = bsr32__full_a ] && continue
    h=$d/sim2/history.csv
    if [ -f $h ] && tail -1 $h | awk -F, '{exit !($2 > 1e-3 - 1e-9)}'; then runs="$runs $r"; else echo "$set_name $r unfinished"; fi
  done
  # shellcheck disable=SC2086
  $PY $COL runs $P/scale_runs.csv $P/scale_series.csv $set_name $CASE bsr32__full_a bsr32__full_a $runs --prefix $prefix \
      > $P/runs_$set_name.log 2>&1
  echo "runs $set_name rc=$? ($(echo $runs | wc -w) runs)"
done
$PY - $P/scale_runs.csv <<'PYEOF'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    state = sum(float(r[k]) for k in ("bytes_history", "bytes_reference", "bytes_frozen", "bytes_stress")) / 1e6
    tangent = sum(float(r["bytes_%s_device" % c] or 0) for c in ("tangent_values", "tangent_structure", "tangent_work")) / 1e6
    print("%-8s %-32s dofs %-9s elements %-8s steps %-3s failed %-2s tangent %8.1f MB prony %8.1f MB peak %8.0f MB" % (
        r["set"], r["run"], r["dofs"], r["elements"], r["steps"], r["failed"], tangent, state, float(r["device_used_peak"]) / 1e6))
PYEOF
