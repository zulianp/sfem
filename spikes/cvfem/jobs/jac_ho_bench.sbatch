#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-jacho
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=jacho_%j.out
#SBATCH --error=jacho_%j.err

# THE JACOBIAN ACTION, PER CONVECTIVE SCHEME. (paper Figure 9, fig:jacho)
#
# The companion to jobs/conv_ho_bench.sbatch, which measures the same five schemes on the
# RESIDUAL. Everything here is derived from that job -- same passes, same best-of logic, same
# output columns -- so the two figures can be read against each other and a change to the method
# has one place to happen.
#
# Why the action needs its own measurement rather than a scaling of the residual's: the
# correction is DEFERRED, so it contributes nothing to the lagged action, and differentiating it
# costs a reconstruction of the direction's nodal velocity gradient per matvec plus the limiter's
# own derivative. Those costs do not scale with the residual's, and the limiter derivatives differ
# from the limiters -- two of the three need a rounding band where the limiter itself does not
# (see cvfem_venkata_limiter.hpp).
#
# ho_unlim_lagged is a control, not a scheme: the lagged action should equal the first-order
# action exactly, because the correction it omits is the only difference between them.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_jacho/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
CSV=$OUT/jacho.csv

N=${N:-128}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-3}

echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')"
echo "### the JACOBIAN ACTION per convective scheme -- the companion to conv_ho_bench.sbatch"
echo "### ho_unlim_lagged is the lagged action, which carries no correction derivative at all"
echo

printf '%-22s %-8s %10s %10s %10s %10s %14s\n' arm layout MDOF_s dofs elems flops_el checksum
for arm in "first_order|atomic|--jac-action" \
           "first_order|packed|--jac-action" \
           "ho_unlimited|atomic|--jac-action --conv-ho 0" \
           "ho_unlimited|packed|--jac-action --conv-ho 0" \
           "ho_unlim_lagged|packed|--jac-action --conv-ho 0 --lagged-ho" \
           "ho_clip|atomic|--jac-action --conv-ho 1" \
           "ho_clip|packed|--jac-action --conv-ho 1" \
           "ho_venkatakrishnan|atomic|--jac-action --conv-ho 2" \
           "ho_venkatakrishnan|packed|--jac-action --conv-ho 2" \
           "ho_darwish_moukalled|atomic|--jac-action --conv-ho 3" \
           "ho_darwish_moukalled|packed|--jac-action --conv-ho 3"; do
    IFS='|' read -r lbl lay extra <<<"$arm"
    best=""; dofs=""; ck=""; elems=""; fpe=""
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
        # The work model and the element count are pass-invariant, so they may be read from any
        # pass; the roofline multiplies them by the BEST rate, which is the rate this table
        # reports. Reading a GFLOP/s line instead would bind the point to whichever pass it came
        # from and disagree with the MDOF/s beside it.
        elems=$(val 'elements'); fpe=$(val 'flops_per_element_model')
    done
    printf '%-22s %-8s %10s %10s %10s %10s %14s\n' "$lbl" "$lay" "${best:--}" "${dofs:--}" \
           "${elems:--}" "${fpe:--}" "${ck:--}"
done

echo
echo "The checksums must DIFFER between first order and each higher-order arm. For the ACTION the"
echo "reason is sharper than for the residual: the correction is deferred, so an action that was"
echo "not differentiated through it is EXACTLY the first-order action -- an identical checksum"
echo "means the exact term never reached the kernel and the row is first order measured twice."
echo "ho_unlim_lagged is the arm that SHOULD match first_order, and is included as that control."
echo
echo "csv: $CSV"
