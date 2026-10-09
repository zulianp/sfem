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
# the matrix's favour. Lagging drops that term from the matrix-free side, giving the same
# operator the matrix holds. Both are reported: the exact action is what a Newton-Krylov solve
# should apply, the lagged one is what the matrix can offer.
#
# EVERY ARM IS FULLY EXACT OR FULLY LAGGED, never a mixture. The driver has two independent
# switches -- --lagged-rc for the Rhie-Chow pressure-gradient sensitivity and --lagged-ho for the
# deferred correction's velocity-gradient one -- and this job previously lagged the first while
# differentiating the second, which is an operator nobody would choose: exact in one sensitivity
# and not the other, the Jacobian of no residual at all.
#
# One consequence is worth stating because it shapes the table. A FULLY LAGGED ACTION DOES NOT
# DEPEND ON THE CONVECTIVE SCHEME: lagging the correction makes it constant in the direction, so
# it leaves the action entirely, and lagging the Rhie-Chow term removes the only other
# direction reconstruction there is. What is left is the same
# operator whatever --conv-ho says. So there is ONE lagged row rather than one per scheme, and
# The controls below are what prove it: the same flags plus --conv-ho at EVERY limiter, each of
# whose fingerprints must equal jac_lagged packed's bit for bit. One limiter is not enough for a
# row that claims "any scheme", even though the code argument covers all of them -- the apply
# call passes `with_ho ? &hop : nullptr` and the limiter id lives on hop, so with the correction
# lagged the scheme has no route into the kernel at all (src/best/cvfem_hex8_best_packed.hpp,
# apply_jacobian_action_packed; the atomic and element-coloured twins do the same). Measured
# anyway, because that is a claim about five configurations and the table makes it about five.
#
# Packed only, necessarily: the atomic sweep fixes no summation order, so it has no fingerprint
# to compare and the structural argument is all there is for its column.
#
# The exact arms span the schemes, because that is what the matrix gives up. A higher-order
# correction is an operator the matrix cannot hold at any sparsity: differentiating it reaches
# the direction's nodal velocity gradient, which widens the coupling past the one ring the matrix
# is built on. Two limiters rather than one -- Barth-Jespersen is the cheapest of them and
# Darwish-Moukalled among the dearest -- so the pair brackets what a limiter costs the action.
#
# Best of three passes per arm, for the run-to-run band this benchmark has already been shown
# to carry.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
# THE SINGLE-PRECISION BUILD, in the same allocation, every pass running both binaries with the
# order alternating (see jobs/conv_ho_bench.sbatch). In it every vector and the matrix are float,
# so its one SpMV arm is the fully single-precision product -- the like-for-like reference for the
# f32 matrix-free rows -- and spmv_f64 is not run there: --bsr-precision double would store float
# values under a double label.
B32=$PWD/build/cvfem_hex8_ns_upwind_bench_f32
PRECS="64"; [ -x "$B32" ] && PRECS="64 32"
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
printf '%-26s %-8s %10s %10s %12s %12s %5s\n' operator layout MDOF_s dofs qgrad_frac hograd_frac prec
declare -A FP
for arm in "jac_exact|packed|--jac-action --rhie-chow" \
           "jac_exact|atomic|--jac-action --rhie-chow" \
           "jac_exact|ecolor|--jac-action --rhie-chow" \
           "jac_clip_exact|packed|--jac-action --rhie-chow --conv-ho 1" \
           "jac_clip_exact|atomic|--jac-action --rhie-chow --conv-ho 1" \
           "jac_clip_exact|ecolor|--jac-action --rhie-chow --conv-ho 1" \
           "jac_ho_exact|packed|--jac-action --rhie-chow --conv-ho 3" \
           "jac_ho_exact|atomic|--jac-action --rhie-chow --conv-ho 3" \
           "jac_ho_exact|ecolor|--jac-action --rhie-chow --conv-ho 3" \
           "jac_lagged|packed|--jac-action --rhie-chow --lagged-rc --lagged-ho" \
           "jac_lagged|atomic|--jac-action --rhie-chow --lagged-rc --lagged-ho" \
           "jac_lagged|ecolor|--jac-action --rhie-chow --lagged-rc --lagged-ho" \
           "jac_lagged_ctl_unlim|packed|--jac-action --rhie-chow --lagged-rc --lagged-ho --conv-ho 0" \
           "jac_lagged_ctl_clip|packed|--jac-action --rhie-chow --lagged-rc --lagged-ho --conv-ho 1" \
           "jac_lagged_ctl_venk|packed|--jac-action --rhie-chow --lagged-rc --lagged-ho --conv-ho 2" \
           "jac_lagged_ctl_dm|packed|--jac-action --rhie-chow --lagged-rc --lagged-ho --conv-ho 3" \
           "spmv_f64|none|--bsr-apply --bsr-precision double" \
           "spmv_f32|none|--bsr-apply --bsr-precision single"; do
    IFS='|' read -r lbl lay extra <<<"$arm"
    # "none" rather than "n/a": the label reaches a filename and a slash there
    # silently redirected into a non-existent directory.
    layflag=""; [ "$lay" != "none" ] && layflag="--layout $lay"
    [ -n "$PACK" ] && [ "$lay" = "packed" ] && layflag="$layflag --pack-size $PACK"
    declare -A best=() dofs=() qg=() hg=()
    for pass in $(seq 1 "$PASSES"); do
        order=$PRECS; [ $((pass % 2)) -eq 0 ] && order=$(echo $PRECS | awk '{for(i=NF;i>0;i--) printf "%s ", $i}')
        for prec in $order; do
            [ "$prec" = 32 ] && [ "$lbl" = spmv_f64 ] && continue
            bin=$B; [ "$prec" = 32 ] && bin=$B32
            log=$OUT/${lbl}_${lay}_f${prec}_p${pass}.log
            timeout "${RUN_TIMEOUT:-360}" \
              srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
              bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $bin --n $N --repeat $REPEAT --warmup 3 $layflag $extra --csv $CSV --tag ${lbl}_${lay}_f${prec}_p${pass}" \
              > "$log" 2>&1
            val() { sed -n "s|^  $1: \(.*\)$|\1|p" "$log" | head -1; }
            v=$(val 'MDOF/s'); [ -n "$v" ] || continue
            best[$prec]=$(awk -v a="${best[$prec]:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
            # frac_jac_action_qgrad / _hograd, which is what the bench prints. The job read
            # 'frac_qgrad' -- the CSV's column name, not the stdout line's -- so the column was "-"
            # for every arm and the reconstruction's share of each matvec was never recorded. The two
            # together are what says why lagging the Rhie-Chow term buys so much less under a
            # higher-order scheme: when both direction gradients are needed they are ONE fused
            # four-field sweep, charged once to hograd, so dropping the pressure one leaves the
            # sweep's fixed half behind instead of removing it.
            dofs[$prec]=$(val 'dofs'); qg[$prec]=$(val 'frac_jac_action_qgrad'); hg[$prec]=$(val 'frac_jac_action_hograd')
            # Kept for the scheme-independence check below. Packed only: the atomic sweep fixes no
            # summation order, so its fingerprint differs run to run and proves nothing either way.
            [ "$lay" = packed ] && FP[$prec|$lbl]=$(grep -am1 'fingerprint:' "$log" | awk '{print $2}')
        done
    done
    for prec in $PRECS; do
        [ "$prec" = 32 ] && [ "$lbl" = spmv_f64 ] && continue
        printf '%-26s %-8s %10s %10s %12s %12s %5s\n' "$lbl" "$lay" "${best[$prec]:--}" \
               "${dofs[$prec]:--}" "${qg[$prec]:--}" "${hg[$prec]:--}" "f$prec"
    done
done
echo

# THE TABLE HAS ONE LAGGED ROW, AND THIS IS WHY IT MAY.
for prec in $PRECS; do
ref=${FP[$prec|jac_lagged]:-}
verdict=YES
if [ -z "$ref" ]; then
    verdict="UNKNOWN (the reference arm produced no fingerprint)"
else
    for c in unlim clip venk dm; do
        got=${FP[$prec|jac_lagged_ctl_$c]:-}
        if [ -z "$got" ]; then
            verdict="UNKNOWN (control $c produced no fingerprint)"
        elif [ "$got" != "$ref" ]; then
            verdict="NO -- control $c gave $got against $ref; the single lagged row is WRONG"
        fi
        printf '  f%s control %-6s %s %s\n' "$prec" "$c" "${got:--}" \
               "$([ "$got" = "$ref" ] && echo match || echo DIFFERS)"
    done
fi
echo "f$prec: fully lagged is scheme-independent across every limiter: $verdict${ref:+ ($ref)}"
done
echo
echo "The exact arms carry the direction-gradient pass; jac_lagged does not, and is the operator"
echo "the assembled matrix encodes. Compare spmv_* against jac_lagged for like-for-like."
echo "jac_lagged_ctl_* are controls, not rows: each must reproduce jac_lagged packed exactly."
echo "csv: $CSV"
