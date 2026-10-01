#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-hosize
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=hosize_%j.out
#SBATCH --error=hosize_%j.err

# THE HIGHER-ORDER RESIDUAL ACROSS PROBLEM SIZE. (paper Figure 5)
#
# Figure 5 plots throughput against the cube edge for the two layouts and the lagged product.
# Every series in it was first order, so the figure said nothing about whether the format's
# margin holds as the scheme gets richer -- which is the one thing the convective sweep at a
# single size cannot show. This adds the limited higher-order arm at the same sizes.
#
# Darwish--Moukalled, matching Figure 7 and the fair-comparison table, and carrying Rhie--Chow
# as those do, so the arm is one of the operators the paper already reports rather than a sixth
# configuration.
#
# Two passes rather than three: ten arms at five sizes, and n=192 builds a 7-million-element
# mesh. The pack-size sweep measured the run-to-run band this guards against as large on a
# single pass, so two is the minimum that means anything; if the debug window is tight, drop a
# size rather than a pass.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_hosize/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
SIZES=${SIZES:-"64 96 128 160 192"}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-2}
PACK=${PACK:-1024}

echo "### host $(hostname)  repeat=$REPEAT passes=$PASSES pack=$PACK  $(date '+%F %T')"
echo "### residual, --rhie-chow --conv-ho 3, against the first-order arm at the same sizes"
echo
printf '%-8s %-8s %-10s %12s %12s\n' n layout scheme MDOF_s dofs

for n in $SIZES; do
  for lay in packed atomic; do
    layflag="--layout $lay"
    [ "$lay" = packed ] && layflag="$layflag --pack-size $PACK"
    for sch in "fo|--rhie-chow" "ho|--rhie-chow --conv-ho 3"; do
      IFS='|' read -r tag extra <<<"$sch"
      best=""; dofs=""
      for pass in $(seq 1 "$PASSES"); do
        log=$OUT/n${n}_${lay}_${tag}_p${pass}.log
        timeout "${RUN_TIMEOUT:-420}" \
          srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                    $B --n $n --repeat $REPEAT --warmup 3 $layflag --kernel sumfact $extra" \
          > "$log" 2>&1
        val() { sed -n "s|^  $1: \(.*\)$|\1|p" "$log" | head -1; }
        v=$(val 'MDOF/s'); [ -n "$v" ] || continue
        best=$(awk -v a="${best:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
        dofs=$(val 'dofs')
      done
      printf '%-8s %-8s %-10s %12s %12s\n' "$n" "$lay" "$tag" "${best:--}" "${dofs:--}"
    done
  done
done
echo "### done"
