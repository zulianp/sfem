#!/bin/bash
#SBATCH --job-name=collect-dt
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/collect_dt_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The step-size study of the strokes on TET4 (tangent9_valves.sbatch with EXTRA="--set time.dt=<dt>
# --set output.export_freq=<0.1 ms / dt>", PREFIX dt<us>_s0; dt 100 us is the campaign's la_s0 (LDP24A)
# or tp10_s0 (FF200);
# below 0.313 us the sixth term stores a tensor):
# per step, the reference twice (bsr32__full_a, _b) and t* with int16 remainders and float32 slots.
# Into OUT (a new folder; the script stops if it exists): every run against the finest step's
# reference (set dt_TET4), every run of a step against that step's reference (sets dt_own_<us>:
# the repeat's floor and each storage's error at that step), and the valve events of every run.
# Runs a step lacks are left out.
# MESH names the folders' mesh part, s0 (TET4) or p2 (TET10). The defaults are FF200's campaign;
# LDP24A's (collect_la_studies.sbatch) passes CASE, TAGS (first la) and CHECK, the campaign
# collection's valve_runs.csv (collect9.sbatch PUMP=LDP24A), which a TET4 study needs to exist.
#   [CASE=ldp_cases/FF200_ExodusMesh] [TAGS="tp10 dt050 dt025 dt010 dt005 dt0025 dt00125 dt000625 dt0003125 dt00015625"]
#   [MESH=s0] [OUT=runs/paper_dt_<finest tag>] [CHECK=runs/paper_data10/valve_runs.csv]
set -u
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
PY=$SFEM_PY
COL=$PAPER_PY/collect_prony.py
CASE=${CASE:-$LDP_CASES/FF200_ExodusMesh}
TAGS=${TAGS:-tp10 dt050 dt025 dt010 dt005 dt0025 dt00125 dt000625 dt0003125 dt00015625}
CHECK=${CHECK:-$SFEM_RUNS/paper_data10/valve_runs.csv}
MESH=${MESH:-s0}
[ $MESH = s0 ] && [ ! -f "$CHECK" ] && { echo "collect_dt: no $CHECK to check the campaign's repeat against; give CHECK"; exit 1; }
SET=dt_TET4; [ $MESH = p2 ] && SET=dt_TET10
finest=${TAGS##* }
P=${OUT:-$SFEM_RUNS/paper_dt_$finest}
[ -e $P ] && { echo "collect_dt: $P exists; give a new OUT"; exit 1; }
mkdir -p $P
RUNS=${RUNS:-bsr32__full_a bsr32__full_b qbsr16__int16_scaled_s32_a qbsr16__full_a bsr32__int16_scaled_s32_a bsr32__full_s32_a bsr32__float32_plain_s32_a}
all=""
for tag in $TAGS; do for r in $RUNS; do [ -f $CASE/${tag}_${MESH}_$r/sim2/history.csv ] && all="$all ${tag}_${MESH}_$r"; done; done
ref=${finest}_${MESH}_bsr32__full_a
others=$(for r in $all; do [ $r != $ref ] && echo $r; done)
# shellcheck disable=SC2086
$PY $COL runs $P/dt_runs.csv $P/dt_series.csv $SET $CASE $ref $others > $P/dt.log 2>&1; echo "runs rc=$?"
for tag in $TAGS; do
  own=$(for r in $all; do case $r in (${tag}_${MESH}_*) [ $r != ${tag}_${MESH}_bsr32__full_a ] && echo $r ;; esac; done)
  # shellcheck disable=SC2086
  $PY $COL runs $P/dt_runs.csv $P/dt_own_series.csv dt_own_$tag $CASE ${tag}_${MESH}_bsr32__full_a $own \
      >> $P/dt.log 2>&1; echo "own $tag rc=$?"
done
# shellcheck disable=SC2086
$PY $COL events $P/dt_events.csv $SET $CASE $all >> $P/dt.log 2>&1; echo "events rc=$?"
# The collector reads the frames' times from out/time.txt: the campaign's TET4 repeat must match valve_runs.csv.
camp=${TAGS%% *}
if [ $MESH = s0 ]; then
  $PY $COL runs $P/check_runs.csv $P/check_series.csv TET4 $CASE ${camp}_s0_bsr32__full_a ${camp}_s0_bsr32__full_b >> $P/dt.log 2>&1
  $PY - $P/check_runs.csv $CHECK ${camp}_s0_bsr32__full_b <<'PYEOF'
import csv, sys
new = {r["run"]: r for r in csv.DictReader(open(sys.argv[1]))}[sys.argv[3]]
old = {(r["set"], r["run"]): r for r in csv.DictReader(open(sys.argv[2]))}[("TET4", "bsr32__full_b")]
print("check e_u median", new["e_u_median"], "against", old["e_u_median"])
PYEOF
fi
$PY - $P/dt_runs.csv <<'PYEOF'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    print("%-14s %-36s steps %-6s failed %-2s e_u med %-10.3g max %-10.3g e_inf max %-10.3g stored %s time %s" % (
        r["set"], r["run"], r["steps"], r["failed"], float(r["e_u_median"]), float(r["e_u_max"]), float(r["e_inf_max"]),
        r["iso_stored"], r["stepper_s"]))
PYEOF
