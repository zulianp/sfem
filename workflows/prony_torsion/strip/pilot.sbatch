#!/bin/bash
#SBATCH --job-name=strip-pilot
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/strip_pilot_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The strip wound by full turns (cases/epdm_strip_turns.yaml), a pilot: does Newton carry N turns, the
# hold and the release, and how long does a run take on one Grace (72 cores)? Four runs at once:
# HEX8_60 and TET10 (TET4_60 promoted) at N = 1 and 2, history in double.
#   [BUILD=<host build dir>] [OUT=<folder>]
set -u
SRC=/ritom/scratch/cscs/zulianp/sfem-prony-md
BUILD=${BUILD:-$SRC/build-x27}
OUT=${OUT:-$SRC/runs/strip_pilot}
CASE=$SRC/workflows/prony_torsion/cases/epdm_strip_turns.yaml
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
go() {  # <mesh> <turns> [options]
  local mesh=$1 n=$2; shift 2
  local W=$OUT/${mesh}_N$n
  rm -rf $W; mkdir -p $W; cd $W
  cp $CASE case.yaml; ln -s $OUT/meshes/${mesh%%-*} mesh
  local start=$(date +%s.%N)
  SMESH_TRACE_FILE=$W/smesh.trace.csv srun --exact -n 1 -c 72 $BUILD/playground case.yaml \
      --set constraints.0.angle=$(echo "$n * 6.283185307179586" | bc -l) "$@" > run.log 2>&1
  echo "$mesh N=$n rc=$? wall=$(echo "$(date +%s.%N) - $start" | bc)s steps=$(grep -cE '^step .*converged' run.log)" \
       "failed=$(grep -c 'attempt failed' run.log) t=$(tail -1 results/history.csv | cut -d, -f2)"
}
mkdir -p $OUT
[ -d $OUT/meshes ] || $SRC/venv/bin/python $SRC/workflows/prony_torsion/strip/make_meshes.py $OUT/meshes
go HEX8_60 1 & go HEX8_60 2 &
go TET4_60-p2 1 --set mesh.promote_to_p2=true & go TET4_60-p2 2 --set mesh.promote_to_p2=true &
wait
