#!/bin/bash
#SBATCH --job-name=collect-la-studies
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/collect_la_studies_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The LDP paper's studies on LDP24A_r100 (tangent9_valves.sbatch's runs with the prefixes below), each into a new folder (every
# collector stops if its folder exists): the step study on TET4 (collect_dt.sbatch, OUT/dt) and on
# TET10 over the first stroke (OUT/dt10), the series cut to three terms at 5 us (OUT/tk:
# dt_fourterm_runs.csv, its two runs against each other; dt_cross_runs.csv, against the full
# series), the updates int16 rounds away (OUT/dg/dt_diag.csv, the last history row of each dg run)
# and the storage at eight times the elements (collect_scale.sbatch, OUT/scale).
# The TET4 step study checks the campaign's repeat against CHECK, the campaign collection's
# valve_runs.csv (collect9.sbatch PUMP=LDP24A); it must exist.
#   [OUT=runs/paper_la_studies] [CHECK=runs/paper_la/valve_runs.csv]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
PY=$D/venv/bin/python
R=$D/workflows/ldp_containment/transient/runs
COL=${PAPER_PY:-/ritom/scratch/cscs/zulianp/xsesfsi_papers/ldp_containment/python}/collect_prony.py
CASE=/ritom/scratch/cscs/zulianp/ldp_cases/LDP24A_r100
O=${OUT:-$D/runs/paper_la_studies}
CHECK=${CHECK:-$D/runs/paper_la/valve_runs.csv}
[ -e $O ] && { echo "collect_la_studies: $O exists; give a new OUT"; exit 1; }
[ -f $CHECK ] || { echo "collect_la_studies: no $CHECK (collect9.sbatch PUMP=LDP24A); give CHECK"; exit 1; }
mkdir -p $O
CASE=$CASE TAGS="la dt050 dt025 dt010 dt005 dt0025 dt00125" CHECK=$CHECK OUT=$O/dt bash $R/collect_dt.sbatch > $O/dt.out 2>&1; echo "dt rc=$?"
CASE=$CASE TAGS="la dw025 dw010 dw005 dw0025" MESH=p2 RUNS="bsr32__full_a bsr32__full_b qbsr16__full_a bsr32__int16_scaled_s32_a" \
    OUT=$O/dt10 bash $R/collect_dt.sbatch > $O/dt10.out 2>&1; echo "dt10 rc=$?"
mkdir -p $O/tk
$PY $COL runs $O/tk/dt_fourterm_runs.csv $O/tk/dt_fourterm_series.csv dt_own_tk005 $CASE tk005_s0_bsr32__full_a tk005_s0_bsr32__full_b \
    > $O/tk/tk.log 2>&1; echo "tk own rc=$?"
$PY $COL runs $O/tk/dt_cross_runs.csv $O/tk/dt_cross_series.csv cross_005 $CASE dt005_s0_bsr32__full_a tk005_s0_bsr32__full_a \
    >> $O/tk/tk.log 2>&1; echo "tk cross rc=$?"
mkdir -p $O/dg
$PY - $CASE $O/dg/dt_diag.csv <<'PYEOF'
import csv, os, sys
case, out = sys.argv[1:]
rows = []
for tag, dt in (("100", 100), ("010", 10), ("0025", 2.5)):
    h = os.path.join(case, "dg%s_s0_bsr64__int16_scaled_plain_s32_a" % tag, "sim2", "history.csv")
    if not os.path.exists(h):
        print("dg%s: no history" % tag)
        continue
    last = list(csv.DictReader(open(h)))[-1]
    rows.append({"dt_us": "%g" % dt, "t": last["t"], "iso_stored": last["iso_stored"], "updates": last["updates"],
                 "updates_lost": last["updates_lost"]})
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dt_us", "t", "iso_stored", "updates", "updates_lost"])
    w.writeheader()
    w.writerows(rows)
print(rows)
PYEOF
echo "dg rc=$?"
# TET10x8: one step (1e-4 s) in sc1_p2.
CASE=$CASE TAG=sc TAG10=sc1 T_END10=1e-4 OUT=$O/scale bash $R/collect_scale.sbatch > $O/scale.out 2>&1; echo "scale rc=$?"
tail -n 40 $O/dt.out $O/dt10.out $O/scale.out
