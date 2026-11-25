#!/bin/bash
#SBATCH --job-name=strip-tight
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/strip_tight_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The Newton tolerance's own effect on the strip (campaign.sbatch's case, one turn, TET10_60): double,
# int16 and float16 a.+d. with remainders again at 1e-9 N instead of 1e-6 N (<run>_tight, 60 Newton
# iterations), then e_u of the campaign's runs and of these against the tight double run
# (set strip_tolerance) into P/strip_tolerance_{runs,series}.csv (P: runs/paper_strip<tag>; the job
# stops if they exist). A run whose folder exists is kept. BUILD as campaign.sbatch's.
# STRIP_TAG and EXTRA as campaign.sbatch's (EXTRA: split by python's shlex, needs a STRIP_TAG, must
# match runs/strip<tag>/extra.txt).
#   [MESH=TET10_60] [BUILD=build-x28] [P=runs/paper_strip<tag>] [STRIP_TAG=] [EXTRA=]
set -u
[ -n "${EXTRA:-}" ] && [ -z "${STRIP_TAG:-}" ] && { echo "tight: EXTRA needs a STRIP_TAG"; exit 1; }
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
D=$SFEM_ROOT
MESH=${MESH:-TET10_60}
B=${BUILD:+$D/$BUILD}; B=${B:-${SFEM_BUILD:-$D/build-x28}}
src=$MESH; extra=""
[ $MESH = TET10_60 ] && { src=TET4_60; extra="--set mesh.promote_to_p2=true"; }
. "$D/workflows/prony_torsion/strip/runs.sh"
strip_extra "$SFEM_RUNS/strip${STRIP_TAG:-}" || exit 1
OUT=$SFEM_RUNS/strip${STRIP_TAG:-}/$MESH
P=${P:-$SFEM_RUNS/paper_strip${STRIP_TAG:-}}
[ -e $P/strip_tolerance_runs.csv ] && { echo "tight: $P/strip_tolerance_runs.csv exists; give a new P"; exit 1; }
go() {  # <run> <format>
  local W=$OUT/$1_tight
  [ -e $W ] && { echo "$MESH $1_tight: $W exists; kept"; return; }
  mkdir -p $W; cd $W || return
  ln -s $SFEM_RUNS/strip/meshes/$src mesh; cp $D/workflows/prony_torsion/cases/epdm_strip_turns.yaml case.yaml
  SMESH_TRACE_FILE=$W/smesh.trace.csv $SFEM_LAUNCH_HOST $B/playground case.yaml --set execution_space=host \
      --set material.relaxation.iso.history_storage=$2 --set solver.newton.atol=1e-9 --set solver.newton.max_it=60 \
      $extra "${opts[@]}" > run.log 2>&1
  echo "$MESH $1_tight rc=$? failed=$(grep -c 'attempt failed' run.log) t=$(tail -1 results/history.csv | cut -d, -f2)"
}
go full_a full &
go int16_scaled int16_scaled &
go float16_md float16_md &
wait
mkdir -p $P
$SFEM_PY $PAPER_PY/collect_prony.py runs $P/strip_tolerance_runs.csv $P/strip_tolerance_series.csv \
    strip_tolerance $OUT full_a_tight full_a full_b int16_scaled int16_scaled_tight float16_md float16_md_tight --sub results \
    > $P/strip_tolerance.log 2>&1
echo "collect rc=$?"
