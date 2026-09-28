#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-ab-fo
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=abfo_%j.out
#SBATCH --error=abfo_%j.err

# THE FIRST-ORDER PACKED RESIDUAL, BEFORE AGAINST AFTER, IN ONE ALLOCATION.
#
# The standing performance gate is the packed matrix-free throughput, and the higher-order work
# changed a header the first-order kernel also compiles through -- a template parameter on the
# face kernel, a flatten attribute on it, and a shape-function table. None of those should reach
# the HO=false instantiation, but "should" is not a measurement, and a three-node spread of 1.8%
# is not evidence either way.
#
# Interleaved, not blocked: the two binaries alternate within each pass, so a drift in clock or
# in a neighbour's interference lands on both arms instead of on whichever ran second. Best of
# three per arm, because a single pass on this kernel has been measured up to 58% apart.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
BEFORE=${BEFORE:-/ritom/scratch/cscs/zulianp/cvfem_ab/before/build/cvfem_hex8_ns_upwind_bench}
AFTER=${AFTER:-$PWD/build/cvfem_hex8_ns_upwind_bench}
N=${N:-128}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-3}
OUT=${OUT:-$SCRATCH/cvfem_abfo/$SLURM_JOB_ID}; mkdir -p "$OUT"
# The image only: init.sh's line carries --view as well, and passing the whole string to
# srun --uenv= is rejected with "invalid uenv description".
U=$(grep '^uenv start' ~/init.sh | awk '{print $NF}')

echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')"
echo "### before: $BEFORE"
echo "### after:  $AFTER"
echo

run_one() {  # run_one <label> <binary> <pass> <extra...>
    local lbl=$1 bin=$2 pass=$3; shift 3
    local log=$OUT/${lbl}_p${pass}.log
    timeout "${RUN_TIMEOUT:-360}" \
      srun -u --ntasks=1 --cpus-per-task=72 --uenv="$U" --view=default \
      bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                $bin --n $N --repeat $REPEAT --warmup 3 --layout packed --kernel sumfact $*" \
      > "$log" 2>&1
    sed -n 's|^  MDOF/s: \(.*\)$|\1|p' "$log" | head -1
}

declare -A best
for arm in "first_order|" "first_order_rc|--rhie-chow" "jac_action|--jac-action"; do
    IFS='|' read -r lbl extra <<<"$arm"
    for pass in $(seq 1 "$PASSES"); do
        for side in before after; do
            bin=$BEFORE; [ "$side" = after ] && bin=$AFTER
            v=$(run_one "${lbl}_${side}" "$bin" "$pass" $extra)
            [ -n "$v" ] || continue
            k="${lbl}_${side}"
            best[$k]=$(awk -v a="${best[$k]:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
        done
    done
done

printf '%-18s %12s %12s %10s\n' arm before after after/before
for lbl in first_order first_order_rc jac_action; do
    b=${best[${lbl}_before]:-}; a=${best[${lbl}_after]:-}
    r=$(awk -v b="${b:-0}" -v a="${a:-0}" 'BEGIN{print (b+0>0)?sprintf("%.4f", a/b):"-"}')
    printf '%-18s %12s %12s %10s\n' "$lbl" "${b:--}" "${a:--}" "$r"
done
echo
echo "A ratio below 1 is a regression in the standing gate. Node-to-node variation does not"
echo "enter here: both arms ran on $(hostname), interleaved, in this allocation."
