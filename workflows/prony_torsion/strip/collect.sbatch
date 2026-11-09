#!/bin/bash
#SBATCH --job-name=strip-collect
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/strip_collect_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The LDP paper's torsion data from the strip campaign (campaign.sbatch): per mesh every run against
# full_a (strip_runs.csv, strip_series.csv, sets strip_<mesh>) and the diagnostics runs (strip_diag.csv),
# with collect_prony.py, into runs/paper_strip.
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
PY=$D/venv/bin/python
COL=$D/dev/wip/ldp_containment/python/collect_prony.py
P=$D/runs/paper_strip; rm -rf $P; mkdir -p $P
REDUCED="float32 int16_scaled int8_scaled float16_md"
RUNS="full_b"; for f in $REDUCED; do RUNS="$RUNS $f ${f}_plain"; done
RUNS="$RUNS full_a_s32 int16_scaled_s32 float16_md_s32"
for m in HEX8_60 HEX8_120 HEX27_30 HEX27_60 TET4_60 TET10_60; do
  $PY $COL runs $P/strip_runs.csv $P/strip_series.csv strip_$m $D/runs/strip/$m full_a $RUNS --sub results > $P/strip_$m.log 2>&1
  echo "strip $m rc=$?"
done
DRUNS=""; for f in $REDUCED; do DRUNS="$DRUNS $f ${f}_plain"; done
for m in HEX8_120 HEX27_60 TET10_60 TET4_60; do
  [ -d $D/runs/strip_diag/$m ] || continue
  $PY $COL runs $P/strip_diag.csv $P/strip_diag_series.csv diag_$m $D/runs/strip_diag/$m full_a $DRUNS --sub results > $P/diag_$m.log 2>&1
  echo "diag $m rc=$?"
done
wc -l $P/*.csv
