#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-rcjac-packed-ab
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=rcjac_packed_ab_%j.out
#SBATCH --error=rcjac_packed_ab_%j.err
# Packed Rhie-Chow Jacobian action, reference against the Rhie-Chow velocity-derivative change,
# interleaved in one allocation. 8,586,756 dof, 72 threads. BENCH_REF / BENCH_NEW name the binaries.
set -u
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
ARGS="--n 128 --repeat 20 --warmup 3 --layout packed --kernel sumfact --jac-action --rhie-chow"
OUT=${OUT:-$SCRATCH/cvfem_rcjac_packed_ab/$SLURM_JOB_ID}; mkdir -p $OUT
for rep in 1 2 3 4 5 6 7; do
  for nb in ref=$BENCH_REF new=$BENCH_NEW; do
    name=${nb%%=*}; bin=${nb#*=}
    for extra in "" "--boundary"; do
      tag=$name${extra:+_bnd}
      uenv run $UENV_ARGS -- $bin $ARGS $extra > $OUT/${tag}_$rep.log 2>&1
      echo "rep $rep $tag $(grep -E 'MDOF/s_jac_action_kernel_only|^  MDOF/s:' $OUT/${tag}_$rep.log | tr '\n' ' ')"
    done
  done
done
echo "=== DONE $(date)"
