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
# (set strip_tolerance) into runs/paper_strip/strip_tolerance_{runs,series}.csv.
#   [MESH=TET10_60] [BUILD=build-x28]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
MESH=${MESH:-TET10_60}; BUILD=${BUILD:-build-x28}
src=$MESH; extra=""
[ $MESH = TET10_60 ] && { src=TET4_60; extra="--set mesh.promote_to_p2=true"; }
OUT=$D/runs/strip/$MESH
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
go() {  # <run> <format>
  local W=$OUT/$1_tight
  rm -rf $W; mkdir -p $W; cd $W
  ln -s $D/runs/strip/meshes/$src mesh; cp $D/workflows/prony_torsion/cases/epdm_strip_turns.yaml case.yaml
  SMESH_TRACE_FILE=$W/smesh.trace.csv srun --exact -n 1 -c 72 $D/$BUILD/playground case.yaml --set execution_space=host \
      --set material.relaxation.iso.history_storage=$2 --set solver.newton.atol=1e-9 --set solver.newton.max_it=60 \
      --set solver.newton.verbose=true $extra > run.log 2>&1
  echo "$MESH $1_tight rc=$? failed=$(grep -c 'attempt failed' run.log) t=$(tail -1 results/history.csv | cut -d, -f2)"
}
go full_a full &
go int16_scaled int16_scaled &
go float16_md float16_md &
wait
P=$D/runs/paper_strip; mkdir -p $P
rm -f $P/strip_tolerance_runs.csv $P/strip_tolerance_series.csv
$D/venv/bin/python $D/dev/wip/ldp_containment/python/collect_prony.py runs $P/strip_tolerance_runs.csv $P/strip_tolerance_series.csv \
    strip_tolerance $OUT full_a_tight full_a full_b int16_scaled int16_scaled_tight float16_md float16_md_tight --sub results \
    > $P/strip_tolerance.log 2>&1
echo "collect rc=$?"
