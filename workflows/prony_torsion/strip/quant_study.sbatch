#!/bin/bash
#SBATCH --job-name=strip-quant
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-main/runs/strip_quant_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The history's grid and rounding (quant_grid, history_rounding) on the strip wound by one turn
# (cases/epdm_strip_turns.yaml), one mesh per job, against double storage:
#   full_a, full_b             double, twice (the floor)
#   int16, int8                today's per-element scales (int16_scaled, int8_scaled)
#   grid5, grid6, grid7        one power-of-two grid per stored term, tolerance 1e-5, 1e-6, 1e-7
#   <run>_sr                   the same with stochastic rounding
#   <run>_plain                without the remainder form (remainder_horizon 0)
# Reduced formats keep the remainder form otherwise, and double stress slots. One Grace chip
# (72 threads) per run, four at once. Submit one job per mesh:
#   MESH=HEX8_60|HEX8_120|HEX27_30|HEX27_60|TET4_60|TET10_60 [BUILD=build-cpu] [RUNS="..."] [OUT=...]
# Folders $OUT/<mesh>/<run>; a run whose folder exists is skipped, nothing is deleted or overwritten.
set -u
D=/ritom/scratch/cscs/zulianp/sfem-main
BUILD=${BUILD:-build-cpu}
OUT=${OUT:-$D/runs/strip_quant}
MESHES=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/strip/meshes
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
src=$MESH; extra=""
[ $MESH = TET10_60 ] && { src=TET4_60; extra="--set mesh.promote_to_p2=true"; }
G="--set material.relaxation.iso"
options() {  # <run>: the history options of a run name
  local run=$1 a=""
  case $run in *_plain) a="$G.remainder_horizon=0"; run=${run%_plain} ;; esac
  case $run in *_sr) a="$a $G.history_rounding=stochastic"; run=${run%_sr} ;; esac
  case $run in
    full_a | full_b) a="$a $G.history_storage=full" ;;
    int16) a="$a $G.history_storage=int16_scaled" ;;
    int8) a="$a $G.history_storage=int8_scaled" ;;
    grid5) a="$a $G.history_storage=quant_grid $G.history_tolerance=1e-5" ;;
    grid6) a="$a $G.history_storage=quant_grid $G.history_tolerance=1e-6" ;;
    grid7) a="$a $G.history_storage=quant_grid $G.history_tolerance=1e-7" ;;
    *) return 1 ;;
  esac
  echo "$a"
}
go() {  # <run>
  local run=$1 a
  a=$(options $run) || { echo "$run: unknown run"; return; }
  local W=$OUT/$MESH/$run
  [ -e "$W" ] && { echo "$MESH $run: $W exists, skipped"; return; }
  mkdir -p $W; cd $W
  ln -s $MESHES/$src mesh; cp $D/workflows/prony_torsion/cases/epdm_strip_turns.yaml case.yaml
  local start=$(date +%s.%N)
  # shellcheck disable=SC2086
  SMESH_TRACE_FILE=$W/smesh.trace.csv srun --exact -n 1 -c 72 $D/$BUILD/playground case.yaml --set execution_space=host \
      $a $extra --set solver.newton.verbose=true > run.log 2>&1
  echo "$MESH $run rc=$? wall=$(echo "$(date +%s.%N) - $start" | bc)s failed=$(grep -c 'attempt failed' run.log)" \
       "t=$(tail -1 results/history.csv | cut -d, -f2)"
}
RUNS=${RUNS:-"full_a full_b int16 int16_sr int8 int8_sr grid5 grid5_sr grid6 grid6_sr grid7 grid7_sr grid6_plain grid6_sr_plain int16_plain"}
for run in $RUNS; do
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n; done
  go $run &
done
wait
