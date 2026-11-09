#!/bin/bash
#SBATCH --job-name=tangent9
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/tangent9_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The tangent x Prony history campaign of the LDP paper: FF200 simulation 2 (four strokes, BDF2,
# tangent projection on failure), one H100 with 72 cores per run, four at once. Submit with the
# partition and time the element needs (TET4: debug; TET10: normal, --time=00:50:00):
#   ELEMENT=TET4|TET10 RUNS="<run> ..." [SNAP="t0, t1, ..."] [SNAP_RUNS="<run> ..."]
# A run is <tangent>__<history>[_p<P>][_plain][_s32]_<repeat>, its folder
# $CASE/tp9_{s0|p2}_<run>:
#   tangent  bsr64, bsr32 (BSR f64/f32), qbsr16, qbsr8 (stored: bsr f16/i8), em32, em16, em8p
#            (element_matrix f32/f16/i8pair), pt32, pt16, pt8md (points f32/f16/i8md); -np: the
#            tangent projection off
#   history  full, float32, float16, int16_scaled, float16_md; _p<P> elements per
#            mean (TET4), _plain no remainder form, _s32 float32
#            stress slots
# SNAP_RUNS also write the history at the times SNAP (output.history_snapshots). EXTRA: further
# options for every run (a smoke run: EXTRA="--set time.t_end=3e-4"). PREFIX names the folders
# instead of tp9_{s0|p2}, PLAYGROUND another build's binary (an A/B gate, qp_window.sbatch). NO_IC=1
# starts from rest: the run's copy of sim2.yaml loses its initial_conditions (simulation 1's state
# lives on the original mesh; the storage at scale, scale_mesh.sbatch, needs none).
set -u
SRC=/ritom/scratch/cscs/zulianp/sfem-prony-md
PLAYGROUND=${PLAYGROUND:-$SRC/build-cuda/playground}
CASE=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh
T=$SRC/workflows/ldp_containment/transient
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores MPICH_GPU_SUPPORT_ENABLED=0
case ${ELEMENT:-TET4} in
  TET4) SIM1=$CASE/prony3_sim1/sim1; PREFIX=${PREFIX:-tp9_s0}; MESH="" ;;
  TET10) SIM1=$CASE/p2_sim1/sim1; PREFIX=${PREFIX:-tp9_p2}; MESH="--set mesh.path=../db/solid_p2" ;;
  *) echo "unknown ELEMENT ${ELEMENT}"; exit 1 ;;
esac
I="--set solver.linear.format=INEXACT --set solver.linear.inexact.store"
E="--set solver.linear.inexact.encoding"

tangent_args() {  # <tangent>
  local t=${1%-np} a
  case $t in
    bsr64) a="--set solver.linear.format=BSR --set solver.linear.storage_bytes=8" ;;
    bsr32) a="--set solver.linear.format=BSR --set solver.linear.storage_bytes=4" ;;
    qbsr16) a="$I=bsr $E=f16" ;;
    qbsr8) a="$I=bsr $E=i8" ;;
    em32) a="$I=element_matrix $E=f32" ;;
    em16) a="$I=element_matrix $E=f16" ;;
    em8p) a="$I=element_matrix $E=i8pair" ;;
    pt32) a="$I=points $E=f32" ;;
    pt16) a="$I=points $E=f16" ;;
    pt8md) a="$I=points $E=i8md" ;;
    *) return 1 ;;
  esac
  [ "$t" != "$1" ] && a="$a --set solver.newton.tangent_projection=none"
  echo "$a"
}

history_args() {  # <history>[_p<P>][_plain][_s32]
  local h=$1 a="" p=0 horizon=-1 stress=full
  case $h in *_s32) stress=float32; h=${h%_s32} ;; esac
  case $h in *_plain) horizon=0; h=${h%_plain} ;; esac
  case $h in *_p[0-9]*) p=${h##*_p}; h=${h%_p*} ;; esac
  case $h in full | float32 | float16 | int16_scaled | float16_md) ;; *) return 1 ;; esac
  a="--set material.relaxation.iso.history_storage=$h --set material.relaxation.iso.stress_storage=$stress"
  a="$a --set material.relaxation.iso.history_elements_per_mean=$p"
  [ $horizon = 0 ] && a="$a --set material.relaxation.iso.remainder_horizon=0"
  echo "$a"
}

go() {  # <run>
  local run=$1 tangent=${1%%__*} rest=${1#*__} history targs hargs W start
  local snap=()
  history=${rest%_*}
  targs=$(tangent_args "$tangent") || { echo "$run: unknown tangent $tangent"; return; }
  hargs=$(history_args "$history") || { echo "$run: unknown history $history"; return; }
  case " ${SNAP_RUNS:-} " in *" $run "*) snap=(--set "output.history_snapshots={times: [${SNAP}]}") ;; esac
  W=$CASE/${PREFIX}_$run
  rm -rf "$W"; mkdir -p "$W"; cd "$W"
  ln -s "$SIM1" sim1; cp "$T/fp70_membrane_pressure.csv" .; cp "$T/FF200_ExodusMesh/sim2.yaml" .
  # The block from initial_conditions: to the next top-level key, in this run's copy.
  [ "${NO_IC:-0}" = 1 ] && sed -i '/^initial_conditions:/,/^[a-z]/{/^initial_conditions:/d;/^[a-z]/!d}' sim2.yaml
  start=$(date +%s.%N)
  # shellcheck disable=SC2086
  SMESH_TRACE_FILE=$W/smesh.trace.csv srun --exact -n 1 -c 72 --gpus-per-task=1 "$PLAYGROUND" sim2.yaml \
      --set execution_space=device $MESH $targs $hargs ${snap[@]+"${snap[@]}"} ${EXTRA:-} \
      --set output.path=sim2 --set output.history_csv=sim2/history.csv > run.log 2>&1
  local rc=$?
  echo "$ELEMENT $run rc=$rc wall=$(echo "$(date +%s.%N) - $start" | bc)s steps=$(grep -cE '^step .*converged' run.log)" \
       "failed=$(grep -c 'attempt failed' run.log) node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
}

# Sourced (qp_window.sbatch), only the definitions.
[ "${BASH_SOURCE[0]}" != "$0" ] && return 0
for run in ${RUNS}; do
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n; done
  go "$run" &
done
wait
