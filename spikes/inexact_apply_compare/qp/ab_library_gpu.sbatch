#!/bin/bash
#SBATCH --job-name=qp-ab-gpu
#SBATCH --partition=normal
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-pa-lib/runs/%x-%j.out
# The library's device stores against the spike's, on one H100 in one job: the same case, mesh and
# store, the spike's newton_qp_cuda (A) and the library's playground on the device (B), A B B A per
# store so a drift of the node cancels.
#   Args: element case mesh-set     (as gnt.sbatch: stretch_shear | twist, mesh.box.n=[n,n,n] | mesh.path=)
#   STORES: triples spike:store:encoding (default the paper's GPU formats)
# The device applies run asynchronously, so the comparison is on Newton's own totals: the solve and
# its linearisations (each synchronises on the Krylov method's reductions).
A_TREE=/ritom/scratch/cscs/zulianp/sfem-pa-qp
B_TREE=/ritom/scratch/cscs/zulianp/sfem-pa-lib
E=$1; K=$2; M=$3
A=$A_TREE/build-spike-cuda-$E/qp/cuda/newton_qp_cuda
B=$B_TREE/build-cuda/playground
C=$B_TREE/spikes/inexact_apply_compare/qp/cases/$(echo $E | tr A-Z a-z | sed s/proteus_//)_$K.yaml
for x in $A $B; do [ -x $x ] || { echo "missing $x: build on the login node"; exit 1; }; done
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
STORES=${STORES-"f64:points:f64 f16:points:f16 i8md:points:i8md ead16:element_matrix:f16 eadi8p:element_matrix:i8pair bsro16:bsr:f16 bsroi8:bsr:i8"}
case $E in TET4) STORES=${STORES/i8md:points:i8md/} ;; esac
echo "node $(hostname) $E $C $M"
W=$B_TREE/runs/abgpu-$SLURM_JOB_ID; mkdir -p $W
run() {  # run <label> <binary> <args...>
  local label=$1 bin=$2; shift 2
  local dir=$W/$label; mkdir -p $dir; cd $dir
  echo "== $label $*"
  srun --cpu-bind=none -n 1 -c 72 uenv run --view=default prgenv-gnu/24.11:v2 -- $bin $C --set "$M" --set execution_space=device \
    ${DT:+--set time.dt=$DT} "$@" 2>&1 | grep --line-buffered -E "^Playground|newton_qp|^qp:|Abort|rror"
  [ -f smesh.trace.csv ] && grep -E "::inexact_(update|apply)|inexact::make_bsr|BSRTangent::assemble|ConjugateGradient::apply|Newton" smesh.trace.csv | sed 's/^/  trace /'
  cd $W
}
run bsr32-lib $B --set solver.linear.format=BSR --set solver.linear.storage_bytes=4
for s in $STORES; do
  q=${s%%:*}; rest=${s#*:}; store=${rest%%:*}; enc=${rest#*:}
  run $q-spike $A --set solver.linear.format=INEXACT --qp $q
  run $q-lib $B --set solver.linear.format=INEXACT --set solver.linear.inexact.store=$store --set solver.linear.inexact.encoding=$enc
  run $q-lib2 $B --set solver.linear.format=INEXACT --set solver.linear.inexact.store=$store --set solver.linear.inexact.encoding=$enc
  run $q-spike2 $A --set solver.linear.format=INEXACT --qp $q
done
