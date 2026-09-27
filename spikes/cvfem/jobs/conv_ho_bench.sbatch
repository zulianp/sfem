#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-convho
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=convho_%j.out
#SBATCH --error=convho_%j.err

# WHAT THE HIGHER-ORDER CONVECTIVE FLUX COSTS, PER LIMITER ARM. (paper M10)
#
# The deferred-correction flux reconstructs the face value from the donor node and its nodal
# velocity gradient and adds only the DIFFERENCE from first-order upwind, so the correction is
# residual-only and the Jacobian stays first-order. The nodal velocity gradient is hoisted here,
# as the solver lags the correction one Newton step: what this measures is the higher-order
# FLUX, not the reconstruction feeding it.
#
# BOTH LAYOUTS, THROUGH THE SAME KERNEL. The default higher-order sweep is the SCALAR
# sum-factored kernel in both layouts, because on Grace it is the faster of the two packed
# options (see the ho_unlim_simd note below). That is also what makes this a LAYOUT comparison:
# the kernel and its inputs are held fixed and only the gather/scatter differs.
#
# It also means a higher-order packed number is NOT comparable with a first-order packed one,
# which is 16-wide SIMD. The first-order rows below are therefore the atomic ones, so that every
# column of this table is the same kernel; the first-order packed SIMD figure lives with the
# layout campaign and is quoted there.
#
# EVERY PACKED ARM BELOW IS THE GENERATED KERNEL unless its label says _scalar, which forces the
# hand-written one. The _scalar rows are the reference the generated ones are checked against and
# the comparison that says what generating the kernel was worth, arm by arm.
#
# ho_unlim_sympy is the same operator through the GENERATED lane-blocked kernel
# (python/synthesize_cvfem_hex8_ns_upwind_sympy.py): the whole element as one CSE'd block inside
# one lane loop, with no function for a compiler to decline to inline and with the eight node
# coordinates feeding all twelve centroids shared once. Unlimited arm only -- it carries no
# Rhie-Chow and no limiter, and the driver refuses those combinations rather than reporting a row
# that names a term it did not compute.
#
# ho_unlim_simd is the same operator through the SIMD kernel. It is kept as an arm because it
# LOSES: 646.7 MDOF/s scalar against 494.3 SIMD on Grace. The correction needs each lane's 96
# scalars gathered inside the lane loop, and that costs more than vectorising the flux returns.
# The row stays so the result is reproducible rather than remembered.
#
# The two layouts are checked for equivalence by the driver itself
# (verify_packed_ho_residual_vs_atomic_abs, run in one process because packing renumbers the
# mesh); they agree to 1.7e-18 at every limiter arm.
#
# Arms are the limiter ids the solver exposes: 0 unlimited, 1 bounded-face clip,
# 2 Venkatakrishnan, 3 Darwish-Moukalled. The first-order row is the control.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_convho/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
CSV=$OUT/convho.csv

N=${N:-128}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-3}

echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')"
echo "### packed arms are the GENERATED kernel unless labelled _scalar"
echo "### scalar-vs-scalar isolates the LAYOUT; generated-vs-scalar isolates the KERNEL"
echo

printf '%-22s %-8s %10s %10s %14s\n' arm layout MDOF_s dofs checksum
for arm in "first_order|atomic|" \
           "first_order|packed|" \
           "ho_unlimited|atomic|--conv-ho 0" \
           "ho_unlimited|packed|--conv-ho 0" \
           "ho_unlim_scalar|packed|--conv-ho 0 --ho-scalar" \
           "ho_unlim_simd|packed|--conv-ho 0 --ho-simd" \
           "ho_unlim_rc|packed|--conv-ho 0 --rhie-chow" \
           "ho_unlim_rc_scalar|packed|--conv-ho 0 --rhie-chow --ho-scalar" \
           "ho_clip|atomic|--conv-ho 1" \
           "ho_clip|packed|--conv-ho 1" \
           "ho_clip_scalar|packed|--conv-ho 1 --ho-scalar" \
           "ho_venkatakrishnan|atomic|--conv-ho 2" \
           "ho_venkatakrishnan|packed|--conv-ho 2" \
           "ho_venk_scalar|packed|--conv-ho 2 --ho-scalar" \
           "ho_darwish_moukalled|atomic|--conv-ho 3" \
           "ho_darwish_moukalled|packed|--conv-ho 3" \
           "ho_dm_scalar|packed|--conv-ho 3 --ho-scalar"; do
    IFS='|' read -r lbl lay extra <<<"$arm"
    best=""; dofs=""; ck=""
    # Best of three passes: one pass cannot separate this band -- the residual carries a large
    # run-to-run variance that the pack-size sweep measured at up to 58% on a single pass.
    for pass in $(seq 1 "$PASSES"); do
        log=$OUT/${lbl}_${lay}_p${pass}.log
        timeout "${RUN_TIMEOUT:-360}" \
          srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat $REPEAT --warmup 3 --layout $lay --kernel sumfact $extra --csv $CSV --tag ${lbl}_${lay}_p${pass}" \
          > "$log" 2>&1
        val() { sed -n "s|^  $1: \(.*\)$|\1|p" "$log" | head -1; }
        v=$(val 'MDOF/s'); [ -n "$v" ] || continue
        best=$(awk -v a="${best:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
        dofs=$(val 'dofs'); ck=$(val 'checksum')
    done
    printf '%-22s %-8s %10s %10s %14s\n' "$lbl" "$lay" "${best:--}" "${dofs:--}" "${ck:--}"
done

echo
echo "The checksums must DIFFER between first order and each higher-order arm: an identical one"
echo "would mean the correction was not applied and the row is measuring first order twice."
echo
echo "That guard does NOT extend to the Rhie-Chow arms. The checksum is a sum over millions of"
echo "values that is almost entirely cancellation, so a term can be present per node and invisible"
echo "in the total: at n=32 the scalar arm with scale 1.0 prints exactly the no-Rhie-Chow checksum"
echo "while scale 3.0 prints a different one, which is how that was established. Whether the term"
echo "reached the kernel is answered by ran_rc in the CSV and by the driver's per-node oracles"
echo "(--verify-ho --rhie-chow), never by these totals."
echo "csv: $CSV"
