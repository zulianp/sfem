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
# The strip wound by two full turns (cases/epdm_strip_turns.yaml) per history format, on one mesh, for
# the LDP paper's torsion section: two repeats in double, each reduced format with the remainder form
# (<format>) and without it (<format>_plain, remainder_horizon 0), and float32 stress slots (<run>_s32);
# with DIAG=1 the diagnostics build's runs (lost updates) instead. One Grace chip (72 threads) per run,
# four at once. Submit one job per mesh:
#   MESH=HEX8_60|HEX8_120|HEX27_30|HEX27_60|TET4_60|TET10_60 [TURNS=2] [BUILD=build-x27] [DIAG=1]
# Folders $OUT/<mesh>/<run> with OUT=runs/strip (DIAG: runs/strip_diag); meshes from make_meshes.py in
# $OUT/meshes (TET10_60 is TET4_60 promoted by the case).
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
TURNS=${TURNS:-2}
if [ "${DIAG:-0}" = 1 ]; then BUILD=${BUILD:-build-diag}; OUT=$D/runs/strip_diag; else BUILD=${BUILD:-build-x27}; OUT=$D/runs/strip; fi
MESHES=$D/runs/strip/meshes
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
[ -d $MESHES ] || $D/venv/bin/python $D/workflows/prony_torsion/strip/make_meshes.py $MESHES
src=$MESH; extra=""
[ $MESH = TET10_60 ] && { src=TET4_60; extra="--set mesh.promote_to_p2=true"; }
go() {  # <run>
  local run=$1 base=${1%_s32} stress=full fmt h=""
  [ "$base" != "$run" ] && stress=float32
  fmt=${base%_plain}; [ "$fmt" != "$base" ] && h="--set material.relaxation.iso.remainder_horizon=0"
  fmt=${fmt%_[a-d]}
  local W=$OUT/$MESH/$run
  rm -rf $W; mkdir -p $W; cd $W
  ln -s $MESHES/$src mesh; cp $D/workflows/prony_torsion/cases/epdm_strip_turns.yaml case.yaml
  local start=$(date +%s.%N)
  # shellcheck disable=SC2086
  SMESH_TRACE_FILE=$W/smesh.trace.csv srun --exact -n 1 -c 72 $D/$BUILD/playground case.yaml --set execution_space=host \
      --set constraints.0.angle=$(echo "$TURNS * 6.283185307179586" | bc -l) \
      --set material.relaxation.iso.history_storage=$fmt --set material.relaxation.iso.stress_storage=$stress $h $extra \
      --set solver.newton.verbose=true > run.log 2>&1
  echo "$MESH $run rc=$? wall=$(echo "$(date +%s.%N) - $start" | bc)s failed=$(grep -c 'attempt failed' run.log)" \
       "projected=$(grep -c ' projected,' run.log) t=$(tail -1 results/history.csv | cut -d, -f2)"
}
REDUCED="float32 float16_scaled int16_scaled int8_scaled float16_md int8_md"
if [ "${DIAG:-0}" = 1 ]; then
  RUNS="full_a"; for f in $REDUCED; do RUNS="$RUNS $f ${f}_plain"; done
else
  RUNS="full_a full_b"; for f in $REDUCED; do RUNS="$RUNS $f ${f}_plain"; done
  RUNS="$RUNS full_a_s32 int16_scaled_s32 float16_md_s32"
fi
for run in $RUNS; do
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n; done
  go $run &
done
wait
