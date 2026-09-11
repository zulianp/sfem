#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-bench-det
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=bench_determinism_%j.out
#SBATCH --error=bench_determinism_%j.err
#
# Is the bench kernel bit-reproducible with itself? Repeats, not thread counts: one run per
# thread count only asks whether that particular interleaving differed.
#
# Judged on the fingerprint, not the checksum. The checksum is a running sum of millions of
# values, so a last-bit difference on a handful of nodes falls below the ULP of the total and
# rounds away -- it reported the atomic and gathered closures as identical across five
# repeats when a per-node comparison of the same operator showed hundreds of doubles
# differing. The fingerprint folds every value's bit pattern in and changes if any bit does.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
B=$ROOT/build_solver/cvfem_hex8_ns_upwind_bench
OUT=${OUT:-$SCRATCH/cvfem_bench_determinism/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_PROC_BIND=true OMP_PLACES=cores OMP_NUM_THREADS=72
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads 72  $(date)"
echo
printf "%-6s %-6s %-8s %-4s %-26s %s\n" config n mode rep checksum fingerprint
for n in 32 100; do
for cfg in res jac; do
  if [ $cfg = res ]; then FLAGS="--layout packed --rhie-chow --pgrad-per-apply --boundary";
  else FLAGS="--layout packed --jac-action --rhie-chow --boundary"; fi
  for mode in gather atomic; do
    A=$( [ $mode = atomic ] && echo 1 || echo 0 )
    for i in 1 2 3 4 5; do
      log=$OUT/${cfg}_${n}_${mode}_$i.log
      uenv run $U -- env OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
          SFEM_BND_ATOMIC=$A "$B" --n $n --repeat 1 --warmup 0 $FLAGS > "$log" 2>&1
      printf "%-6s %-6s %-8s %-4s %-26s %s\n" "$cfg" "$n" "$mode" "$i" \
          "$(grep -m1 '  checksum:' "$log" | awk '{print $2}')" \
          "$(grep -m1 '  fingerprint:' "$log" | awk '{print $2}')"
    done
  done
done
done
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
