#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-jacfair
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=jacfair_%j.out
#SBATCH --error=jacfair_%j.err

# A LIKE-FOR-LIKE JACOBIAN COMPARISON AGAINST THE ASSEMBLED MATRIX. (paper M11)
#
# The paper compares a matrix-free Jacobian action against an SpMV of the assembled matrix. Left
# as it was, that comparison is not of one operator in two representations: the matrix-free
# action measured is the EXACT Jacobian, which differentiates through the nodal pressure-gradient
# reconstruction and therefore reconstructs the DIRECTION's gradient on every matvec; the
# assembled matrix cannot carry that term at all, because including it would widen the pressure
# coupling past nearest neighbours and break the one-ring sparsity the matrix is built on. The
# matrix lags it.
#
# So the exact action is doing strictly more work than the matrix encodes, and the bias runs in
# the matrix's favour. --lagged-rc drops that term from the matrix-free side, giving the same
# operator the matrix holds. Both are reported: the exact action is what a Newton-Krylov solve
# should apply, the lagged one is what the matrix can offer.
#
# Both convective schemes, not only first-order upwinding. The comparison this job exists to
# make is against an operator the matrix CAN hold, and the higher-order correction is one it
# cannot hold at any sparsity: differentiating it reaches the direction's nodal velocity
# gradient, which widens the coupling past the one ring the matrix is built on. So the
# Two limiters rather than one: the bounded-face clip is the cheapest of them and
# Darwish--Moukalled among the dearest, so the pair brackets what a limiter costs the action.
# higher-order arms have no matrix counterpart, and reporting them beside the first-order ones
# is what shows the size of what the assembled representation gives up -- not a second
# like-for-like row. The scheme is Darwish--Moukalled, the same one Figures 7 to 9 single out.
#
# In the higher-order arms --lagged-rc lags the Rhie--Chow term exactly as it does above; the
# correction itself stays differentiated, because lagging that instead reproduces the
# first-order action to the last bit (paper S7.8) and would duplicate the rows above.
#
# Best of three passes per arm, for the run-to-run band this benchmark has already been shown
# to carry.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_jacfair/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
CSV=$OUT/jacfair.csv
N=${N:-128}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-3}
# The measured optimum for this n, pinned rather than left to the library default. The default
# derives from the index-type ceiling and the thread count, so it moves with the thread count and
# with n, and an arm that takes it is not reproducible against the rest of the paper even when it
# happens to land on the peak -- which at n=128 it does. Read it from the pack-size sweep
# (\packLargeBest); PACK= (empty) restores the library default deliberately.
PACK=${PACK:-1024}

echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')"
echo
printf '%-26s %-8s %10s %10s %12s %12s\n' operator layout MDOF_s dofs qgrad_frac hograd_frac
for arm in "jac_exact|packed|--jac-action --rhie-chow" \
           "jac_exact|atomic|--jac-action --rhie-chow" \
           "jac_lagged|packed|--jac-action --rhie-chow --lagged-rc" \
           "jac_lagged|atomic|--jac-action --rhie-chow --lagged-rc" \
           "jac_clip_exact|packed|--jac-action --rhie-chow --conv-ho 1" \
           "jac_clip_exact|atomic|--jac-action --rhie-chow --conv-ho 1" \
           "jac_clip_lagged|packed|--jac-action --rhie-chow --lagged-rc --conv-ho 1" \
           "jac_clip_lagged|atomic|--jac-action --rhie-chow --lagged-rc --conv-ho 1" \
           "jac_ho_exact|packed|--jac-action --rhie-chow --conv-ho 3" \
           "jac_ho_exact|atomic|--jac-action --rhie-chow --conv-ho 3" \
           "jac_ho_lagged|packed|--jac-action --rhie-chow --lagged-rc --conv-ho 3" \
           "jac_ho_lagged|atomic|--jac-action --rhie-chow --lagged-rc --conv-ho 3" \
           "spmv_f64|none|--bsr-apply --bsr-precision double" \
           "spmv_f32|none|--bsr-apply --bsr-precision single"; do
    IFS='|' read -r lbl lay extra <<<"$arm"
    # "none" rather than "n/a": the label reaches a filename and a slash there
    # silently redirected into a non-existent directory.
    layflag=""; [ "$lay" != "none" ] && layflag="--layout $lay"
    [ -n "$PACK" ] && [ "$lay" = "packed" ] && layflag="$layflag --pack-size $PACK"
    best=""; dofs=""; qg=""; hg=""
    for pass in $(seq 1 "$PASSES"); do
        log=$OUT/${lbl}_${lay}_p${pass}.log
        timeout "${RUN_TIMEOUT:-360}" \
          srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat $REPEAT --warmup 3 $layflag --kernel sumfact $extra --csv $CSV --tag ${lbl}_${lay}_p${pass}" \
          > "$log" 2>&1
        val() { sed -n "s|^  $1: \(.*\)$|\1|p" "$log" | head -1; }
        v=$(val 'MDOF/s'); [ -n "$v" ] || continue
        best=$(awk -v a="${best:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
        # frac_jac_action_qgrad / _hograd, which is what the bench prints. The job read
        # 'frac_qgrad' -- the CSV's column name, not the stdout line's -- so the column was "-"
        # for every arm and the reconstruction's share of each matvec was never recorded. The two
        # together are what says why lagging the Rhie-Chow term buys so much less under a
        # higher-order scheme: when both direction gradients are needed they are ONE fused
        # four-field sweep, charged once to hograd, so dropping the pressure one leaves the
        # sweep's fixed half behind instead of removing it.
        dofs=$(val 'dofs'); qg=$(val 'frac_jac_action_qgrad'); hg=$(val 'frac_jac_action_hograd')
    done
    printf '%-26s %-8s %10s %10s %12s %12s\n' "$lbl" "$lay" "${best:--}" "${dofs:--}" \
           "${qg:--}" "${hg:--}"
done
echo
echo "jac_exact carries the direction-gradient pass; jac_lagged does not, and is the operator"
echo "the assembled matrix encodes. Compare spmv_* against jac_lagged for like-for-like."
echo "csv: $CSV"
