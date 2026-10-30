#!/bin/bash
#SBATCH --job-name=qp-ab-lib
#SBATCH --partition=normal
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-pa-lib/runs/%x-%j.out
# The library's stored-tangent operator against the spike's, on one Grace node in one job: the same
# case, mesh, pack size and store, the spike's newton_qp (A) and the library's playground (B)
# alternated store by store, so node-to-node variation does not enter the comparison.
#   Args: element case mesh-set     (as nt.sbatch: stretch_shear | twist, mesh.box.n=[n,n,n] | mesh.path=)
#   STORES: pairs spike:library (default the paper's eight formats)
# A reads its update and apply totals from its `qp:` line; B from the trace (smesh.trace.csv).
# Each store runs A B B A, so a drift of the node over the job cancels in the sums.
A_TREE=/ritom/scratch/cscs/zulianp/sfem-pa-qp
B_TREE=/ritom/scratch/cscs/zulianp/sfem-pa-lib
E=$1; K=$2; M=$3
A=$A_TREE/build-spike-$E/newton_qp
B=$B_TREE/${B_BUILD:-build}/playground  # B_BUILD=build-native: the library compiled as the spike is (-march=native)
C=$B_TREE/spikes/inexact_apply_compare/qp/cases/$(echo $E | tr A-Z a-z | sed s/proteus_//)_$K.yaml
for x in $A $B; do [ -x $x ] || { echo "missing $x: build on the login node"; exit 1; }; done
case $M in
  mesh.box.n=*) n=${M#*[}; n=${n%%,*}; ne=$((n * n * n)) ;;
  mesh.path=*) ne=$(awk '/n_elements:/{print $2}' ${M#mesh.path=}/meta.yaml) ;;
esac
p=$((ne / 1440 / 16 * 16)); [ $p -gt 8192 ] && p=8192; [ $p -lt 16 ] && p=16
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores SFEM_PACKED_TWO_PASS=1
export SMESH_ELEMENTS_PER_PACK=${SMESH_ELEMENTS_PER_PACK:-$p}
STORES=${STORES-"f64:points:f64 f16:points:f16 i8md:points:i8md ead16:element_matrix:f16 eadi8p:element_matrix:i8pair bsro16:bsr:f16 bsroi8:bsr:i8"}
case $E in TET4) STORES=${STORES/i8md:points:i8md/} ;; esac
echo "node $(hostname) $E $C $M elements $ne packs of $SMESH_ELEMENTS_PER_PACK library $B"
W=$B_TREE/runs/ab-$SLURM_JOB_ID; mkdir -p $W
run() {  # run <label> <binary> <args...>
  local label=$1 bin=$2; shift 2
  local dir=$W/$label; mkdir -p $dir; cd $dir
  echo "== $label $*"
  srun --cpu-bind=none -n 1 -c 72 uenv run --view=default prgenv-gnu/24.11:v2 -- $bin $C --set "$M" ${DT:+--set time.dt=$DT} "$@" 2>&1 \
    | grep --line-buffered -E "^Playground|^step [0-9]|newton_qp|^qp:|Abort|rror"
  [ -f smesh.trace.csv ] && grep -E "::inexact_(update|apply)|BSRTangent::assemble|ConjugateGradient::apply|Newton" smesh.trace.csv | sed 's/^/  trace /'
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
