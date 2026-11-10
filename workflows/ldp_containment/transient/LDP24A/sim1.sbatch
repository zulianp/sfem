#!/bin/bash
#SBATCH --job-name=ldp24a-sim1
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/ldp24a_sim1_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# LDP24A's containment (sim1.yaml) on one GPU per mesh, TET4 and TET10 at once, each in a new folder
# CASE/s1_s0 and CASE/s1_p2 (a run whose folder exists is skipped): the playground's output in sim1/,
# its last state on the whole mesh in sim1/whole (smesh raw_point_data_to_parent), where the strokes
# start. TET10 reads db/solid_p2 (mesh and side-sets). SUFFIX is appended to the folders' names, EXTRA
# passes further options ("--set solver.newton.max_it=60"). The job fails when a run does, so that
# jobs depending on it (afterok) do not start.
#   [CASE=ldp_cases/LDP24A_r100] [ELEMENTS="TET4 TET10"] [SUFFIX=] [EXTRA=]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
CASE=${CASE:-/ritom/scratch/cscs/zulianp/ldp_cases/LDP24A_r100}
T=$D/workflows/ldp_containment/transient/LDP24A
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores MPICH_GPU_SUPPORT_ENABLED=0
go() {  # <element>
  local el=$1 W db
  case $el in TET4) W=$CASE/s1_s0${SUFFIX:-}; db=solid ;; TET10) W=$CASE/s1_p2${SUFFIX:-}; db=solid_p2 ;; *) echo "unknown $el"; return ;; esac
  [ -e $W ] && { echo "$el: $W exists"; return; }
  mkdir -p $W; cd $W
  sed "s#\.\./db/solid/#../db/$db/#g; s#path: \.\./db/solid\$#path: ../db/$db#" $T/sim1.yaml > sim1.yaml
  local start=$(date +%s)
  SMESH_TRACE_FILE=$W/smesh.trace.csv srun --exact -n 1 -c 72 --gpus-per-task=1 $D/build-cuda/playground sim1.yaml \
      --set execution_space=device --set output.path=sim1 --set output.history_csv=sim1/history.csv ${EXTRA:-} > sim1.log 2>&1
  local rc=$?
  $D/venv/bin/python $D/external/smesh/python/smesh/raw_point_data_to_parent.py sim1/out sim1/mesh $CASE/db/$db sim1/whole \
      --fields=disp > map.log 2>&1
  echo "$el rc=$rc map=$? wall=$(( $(date +%s) - start ))s $(tail -1 sim1.log)"
  return $rc
}
pids=()
for el in ${ELEMENTS:-TET4 TET10}; do go $el & pids+=($!); done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
exit $fail
