#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-limiter
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=limiter_%j.out
#SBATCH --error=limiter_%j.err
#
# WHY THE DEFERRED CORRECTION WILL NOT CONVERGE ON THE STEP.
#
# The budget job found it non-convergent at every cell it was run in: 399 Newton steps and
# 103,441 linear iterations at 7,060 dof where first order took 14 and 8,223, and 385 steps
# at 47,268 dof. First order converged in all nine of those cells, so it is the correction
# and not the case. The MMS ladder says the correction is right -- it restored the Re = 100
# rate to ~2 and cut the error 4.9x -- so this is about the iteration, not the scheme.
#
# WHAT THE LOG ACTUALLY SHOWS, which is narrower than "it fails". Stage 1 at Re = 1 converges
# in 7 Newton steps to rel 6.5e-09, as fast as first order. The continuation then bisects its
# way down -- 12 stages -- and stops at Re = 14.04 of a 40 target. In the stage that caps
# out, the residual is not oscillating and not stuck: it falls monotonically by about 20% per
# step, 3.27e-08, 2.57e-08, 2.01e-08, 1.56e-08, and runs out of Newton steps at 1.56e-08.
#
# That is a LINEARLY converging fixed point with a contraction factor near 0.8, which is what
# a lagged term gives when the lag is a large fraction of the operator -- not the chatter a
# switching limiter produces. Both are live hypotheses and they want opposite fixes, so the
# arms are chosen to separate them rather than to confirm either:
#
#   limiter 0   reconstruct, do not limit.      Same crawl => the lag is the cause, and the
#                                               fix is relaxation or a ramp, not a limiter.
#   limiter 1   bounded-face clip.              The failing configuration, reproduced.
#   limiter 2   Venkatakrishnan's smooth bound. Converges => the clip's switching active set
#                                               was the cause after all.
#
# The clip is a real suspect despite the crawl, because Venkatakrishnan (JCP 118, 1995) --
# already quoted in cvfem_hex8_ns_upwind_kernels.hpp for the neighbouring |m| smoothing --
# describes a limiter that "does not settle down" and "inhibits convergence to steady state",
# and states that "if a Newton's method is used ... a differentiable limiter is required".
# That file argued the clip was exempt because the deferred correction keeps it out of the
# Jacobian. True of the Jacobian, false of the iteration: the lag makes Newton a fixed-point
# iteration on the correction as well, and the clip's active set can flip between steps.
#
# final_rel is in the table for exactly this reason. A cell that crawls to 1e-08 and a cell
# that sticks at 1e-02 are both conv=0 and want completely different work.
#
# THE SHARPEST NUMBER IS WHERE IT STALLS, so stall_Re is a column. The budget job's six
# non-converged cells give it away: the continuation stops at the same Reynolds number for
# every target on a given mesh, and the two meshes are an exact factor of four apart.
#
#     n2l4,  7,060 dof:  Re 14.0421 reached, of targets 40, 100 and 200
#     n4l4, 47,268 dof:  Re  3.51052 reached, of targets 40, 100 and 200
#
# Independent of the target means it is a property of the scheme at that Reynolds number and
# not of the ramp that got there. Exactly four apart on a mesh whose h halves means the
# correction gets HARDER to converge as the mesh refines, as h^-2 -- which is backwards for a
# term that is supposed to be a truncation error, and backwards for a cell-Peclet effect too,
# since that would push the stall to a HIGHER Reynolds number on refinement. Whatever is
# growing is not the thing the correction is meant to be.
#
# Two cells, then, not for redundancy but because the ratio between them is the measurement.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "limiter: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_limiter/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

BASE="SFEM_CASE=step SFEM_U=1 SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=vanka SFEM_GMG_OMEGA=1.0"

run_one() {
    local tag=$1; shift
    local log="$OUT/$tag.log"
    local t0=$SECONDS
    stdbuf -oL -eL env $BASE "$@" SFEM_DIAG_CSV="$OUT/$tag.csv" \
        uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    local rc=$? ndof conv nt li en rel
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$log" | head -1)
    nt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    li=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    # The last relative Newton residual the run reached. A cell that did not converge is not
    # reported by its eps_num -- a budget off an unconverged state is not a measurement --
    # but how far it got separates "slow" from "stuck", and those want different fixes.
    rel=$(sed -n 's/.*rel: \([0-9.e+-]*\).*/\1/p' "$log" | tail -1)
    local sre
    sre=$(sed -n 's/.*highest Re SOLVED = \([0-9.e+-]*\) .*/\1/p' "$log" | head -1)
    [ -s "$OUT/$tag.csv" ] && en=$(tail -1 "$OUT/$tag.csv" | cut -d, -f9)
    printf '%-26s %9s %5s %7s %9s %12s %9s %8s %13s\n' \
        "$tag" "${ndof:--}" "${conv:-$rc}" "${nt:--}" "${li:--}" "${rel:---}" "${sre:---}" \
        "$((SECONDS - t0))s" "${en:---}"
}
HDR() { printf '%-26s %9s %5s %7s %9s %12s %9s %8s %13s\n' \
        cell ndof conv newton lin_it final_rel stall_Re wall eps_num; }

HDR
for CELL in "0.05:2:4" "0.05:4:4"; do
    MU=${CELL%%:*}; REST=${CELL#*:}; NN=${REST%%:*}; LV=${REST##*:}
    RE=$(python3 -c "print(int(1*1*2/$MU))")
    # First order, for the reference this is all measured against.
    run_one "re${RE}_n${NN}l${LV}_lo" SFEM_MU=$MU SFEM_N=$NN \
        SFEM_ELEMENT_REFINE_LEVEL=$LV SFEM_CONV_HO=0
    for LM in 0 1 2; do
        run_one "re${RE}_n${NN}l${LV}_ho_lim${LM}" SFEM_MU=$MU SFEM_N=$NN \
            SFEM_ELEMENT_REFINE_LEVEL=$LV SFEM_CONV_HO=1 SFEM_CONV_LIMITER=$LM
    done
    echo
done
echo "logs in $OUT"
echo "done $(date)"
