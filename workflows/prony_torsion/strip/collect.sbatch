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
# with collect_prony.py, into runs/paper_strip. STRIP_TAG: the runs of campaign.sbatch with that tag.
set -u
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
PY=$SFEM_PY
COL=$PAPER_PY/collect_prony.py
P=${OUT:-$SFEM_RUNS/paper_strip}
[ -e $P ] && { echo "collect: $P exists; give a new OUT"; exit 1; }
mkdir -p $P
. "$SFEM_ROOT/workflows/prony_torsion/strip/runs.sh"
RUNS=$(strip_runs | sed 's/^full_a //')
for m in HEX8_60 HEX8_120 HEX27_30 HEX27_60 TET4_60 TET10_60; do
  $PY $COL runs $P/strip_runs.csv $P/strip_series.csv strip_$m $SFEM_RUNS/strip${STRIP_TAG:-}/$m full_a $RUNS --sub results > $P/strip_$m.log 2>&1
  echo "strip $m rc=$?"
done
DRUNS=$(DIAG=1 strip_runs | sed 's/^full_a //')
for m in HEX8_120 HEX27_60 TET10_60 TET4_60; do
  [ -d $SFEM_RUNS/strip_diag${STRIP_TAG:-}/$m ] || continue
  $PY $COL runs $P/strip_diag.csv $P/strip_diag_series.csv diag_$m $SFEM_RUNS/strip_diag${STRIP_TAG:-}/$m full_a $DRUNS --sub results > $P/diag_$m.log 2>&1
  echo "diag $m rc=$?"
done
wc -l $P/*.csv
