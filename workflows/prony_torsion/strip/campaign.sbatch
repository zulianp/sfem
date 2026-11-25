#!/bin/bash
#SBATCH --job-name=strip
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/strip_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The strip wound by one full turn, its end held axially (cases/epdm_strip_turns.yaml), per history
# format, on one mesh, for
# the LDP paper's torsion section: two repeats in double, each reduced format with the remainder form
# (<format>) and without it (<format>_plain, remainder_horizon 0), and float32 evaluated stresses
# (<run>_s32), the lists in runs.sh (the packed grid codes qblock<k>, _sr stochastic rounding);
# with DIAG=1 the diagnostics build's runs (lost updates) instead. One Grace chip (72 threads) per run,
# four at once. Submit one job per mesh:
#   MESH=HEX8_60|HEX8_120|HEX27_30|HEX27_60|TET4_60|TET10_60 [TURNS=1] [BUILD=build-x28] [DIAG=1]
# Two turns with the end held fail on TET10 near 1.3 turns (pilot 5305248).
# Folders $OUT/<mesh>/<run> with OUT=runs/strip (DIAG: runs/strip_diag); meshes from make_meshes.py in
# runs/strip/meshes (TET10_60 is TET4_60 promoted by the case). A run whose folder exists is skipped.
# BUILD names a build folder of the sfem tree; without it, SFEM_BUILD (DIAG: SFEM_DIAG_BUILD), else
# build-x28. The machine: ../../ldp_containment/machine.sh. STRIP_TAG suffixes the run folders
# (runs/strip<tag>, runs/strip_diag<tag>; the meshes stay in runs/strip/meshes), EXTRA adds options to
# every run, e.g. another material. EXTRA is split like a shell command line by python's shlex: quotes
# group words, nothing expands, '#' is no comment, no newlines. It needs a STRIP_TAG (existing runs are
# skipped, so materials would mix); runs/strip<tag>/stamp.txt keeps the tag's TURNS and EXTRA, also for
# DIAG=1, and a job with others stops (runs.sh). Set EXTRA in the environment, not with sbatch --export
# (which splits at commas):
#   STRIP_TAG=_soft EXTRA="--set material.C10=1e5 --set 'material.relaxation.iso.prony=[{g: 0.2, tau: 1}]'"
set -u
[ -n "${EXTRA:-}" ] && [ -z "${STRIP_TAG:-}" ] && { echo "campaign: EXTRA needs a STRIP_TAG"; exit 1; }
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
D=$SFEM_ROOT
TURNS=${TURNS:-1}; MESH=${MESH:-}
. "$D/workflows/prony_torsion/strip/runs.sh"
strip_check || exit 1
if [ "${DIAG:-0}" = 1 ]; then B=${BUILD:+$D/$BUILD}; B=${B:-$SFEM_DIAG_BUILD}; OUT=$SFEM_RUNS/strip_diag${STRIP_TAG:-}
else B=${BUILD:+$D/$BUILD}; B=${B:-${SFEM_BUILD:-$D/build-x28}}; OUT=$SFEM_RUNS/strip${STRIP_TAG:-}; fi
[ -x $B/playground ] || { echo "campaign: no $B/playground"; exit 1; }
strip_stamp || exit 1
MESHES=$SFEM_RUNS/strip/meshes
[ -d $MESHES ] || $SFEM_PY $D/workflows/prony_torsion/strip/make_meshes.py $MESHES
src=$MESH; extra=""
[ $MESH = TET10_60 ] && { src=TET4_60; extra="--set mesh.promote_to_p2=true"; }
go() {  # <run>: <format>[_sr][_plain][_s32] (runs.sh), qblock<k> quant_block at tolerance 1e-<k>
  local run=$1 base=${1%_s32} stress=full fmt h=""
  [ "$base" != "$run" ] && stress=float32
  fmt=${base%_plain}; [ "$fmt" != "$base" ] && h="--set material.relaxation.iso.remainder_horizon=0"
  fmt=${fmt%_[a-d]}
  case $fmt in *_sr) h="$h --set material.relaxation.iso.history_rounding=stochastic"; fmt=${fmt%_sr} ;; esac
  case $fmt in qblock[0-9]) h="$h --set material.relaxation.iso.history_tolerance=1e-${fmt#qblock}"; fmt=quant_block ;; esac
  local W=$OUT/$MESH/$run
  [ -e $W ] && { echo "$MESH $run: $W exists; skipped"; return; }
  mkdir -p $W; cd $W || return
  ln -s $MESHES/$src mesh; cp $D/workflows/prony_torsion/cases/epdm_strip_turns.yaml case.yaml
  local start=$(date +%s.%N)
  # shellcheck disable=SC2086
  SMESH_TRACE_FILE=$W/smesh.trace.csv $SFEM_LAUNCH_HOST $B/playground case.yaml --set execution_space=host \
      --set constraints.0.angle=$(echo "$TURNS * 6.283185307179586" | bc -l) \
      --set material.relaxation.iso.history_storage=$fmt --set material.relaxation.iso.stress_storage=$stress $h $extra \
      "${opts[@]}" > run.log 2>&1
  echo "$MESH $run rc=$? wall=$(echo "$(date +%s.%N) - $start" | bc)s failed=$(grep -c 'attempt failed' run.log)" \
       "projected=$(grep -c ' projected,' run.log) t=$(tail -1 results/history.csv | cut -d, -f2)"
}
RUNS=$(strip_runs)
for run in $RUNS; do
  sfem_throttle
  go $run &
done
wait
