#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-hoexact
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=hoexact_%j.out
#SBATCH --error=hoexact_%j.err

# WHAT THE EXACT HIGHER-ORDER JACOBIAN ACTION COSTS, AGAINST THE LAGGED ONE.
#
# The deferred correction is applied lagged, so it contributes nothing to the Jacobian and the
# action of a higher-order residual is the first-order action. That is a CHOICE: matrix-free, the
# correction can be differentiated, at the price of reconstructing the DIRECTION's nodal velocity
# gradient on every matvec -- nine components per node, against the three the exact Rhie-Chow
# action already rebuilds for the pressure.
#
# Both layouts, because the operator is the same one and a layout comparison that carried the term
# on one side only would report the term as a layout effect. Both arms verified against a finite
# difference of the higher-order residual first (--verify-jac-ho), so a throughput row cannot be
# reported for an operator that does not compute what it claims.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_hoexact/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
CSV=$OUT/hoexact.csv
N=${N:-128}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-3}
# The measured optimum for this n, pinned rather than left to the library default. The default
# derives from the index-type ceiling and the thread count, so it moves with the thread count and
# with n, and an arm that takes it is not reproducible against the rest of the paper even when it
# happens to land on the peak -- which at n=128 it does. Read it from the pack-size sweep
# (\packLargeBest); PACK= (empty) restores the library default deliberately.
PACK=${PACK:-1024}

run() {  # label flags...
    local lbl=$1; shift
    # Only the packed layout has a pack to size; --pack-size on an atomic arm is meaningless.
    local PACKFLAG=""
    case " $* " in *" --layout packed "*) [ -n "$PACK" ] && PACKFLAG="--pack-size $PACK" ;; esac
    local best="" dofs="" hg=""
    for pass in $(seq 1 "$PASSES"); do
        local log=$OUT/${lbl}_p${pass}.log
        timeout "${RUN_TIMEOUT:-420}" \
          srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat $REPEAT --warmup 3 $* $PACKFLAG --csv $CSV --tag ${lbl}_p${pass}" \
          > "$log" 2>&1
        val() { sed -n "s|^  $1: \(.*\)$|\1|p" "$log" | head -1; }
        local v; v=$(val 'MDOF/s'); [ -n "$v" ] || continue
        # Taken from the same pass that gave the best rate, so the fraction and the rate
        # describe one run rather than being averaged across three.
        local faster; faster=$(awk -v a="${best:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?1:0}')
        [ "$faster" = 1 ] && hg=$(val 'frac_jac_action_hograd')
        best=$(awk -v a="${best:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
        dofs=$(val 'dofs')
    done
    printf '%-34s %10s %10s %10s\n' "$lbl" "${best:--}" "${dofs:--}" "${hg:--}"
}

echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')"
echo "### the correction is Venkatakrishnan unless the label says otherwise"
echo

echo "### verification first -- a throughput row for an unverified operator is not a result"
for L in 0 1 2 3; do
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $PWD && OMP_NUM_THREADS=72 $B --n 24 --repeat 1 --warmup 0 --layout packed --jac-action --conv-ho $L --verify-jac-ho" \
      2>&1 | sed -n 's/^verify_jac_ho: /  /p'
done
echo

printf '%-34s %10s %10s %10s\n' arm MDOF_s dofs frac_hograd
run jac_first_order            --layout packed --jac-action --rhie-chow
run jac_ho_lagged_packed       --layout packed --jac-action --rhie-chow --conv-ho 2 --lagged-ho
run jac_ho_exact_packed        --layout packed --jac-action --rhie-chow --conv-ho 2
run jac_ho_lagged_atomic       --layout atomic --jac-action --rhie-chow --conv-ho 2 --lagged-ho
run jac_ho_exact_atomic        --layout atomic --jac-action --rhie-chow --conv-ho 2
run jac_ho_exact_packed_unlim  --layout packed --jac-action --rhie-chow --conv-ho 0
echo
echo "csv: $CSV"
