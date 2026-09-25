#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-peclet
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=peclet_%j.out
#SBATCH --error=peclet_%j.err
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
# THE PECLET SWEEP. One mesh, one geometry, viscosity varied -- so the cell Peclet number
# is the only thing moving and the dissipation reduction can be read against it.
#
#   mu = 0.05  -> Re  40, cell Peclet ~1.7   (the default cell; upwinding barely active)
#   mu = 0.02  -> Re 100, cell Peclet ~4.2
#   mu = 0.01  -> Re 200, cell Peclet ~8.3
#   mu = 0.005 -> Re 400, cell Peclet ~17
#
# Raising Re rather than coarsening is deliberate: it is how a real run reaches high cell
# Peclet, and it keeps the mesh -- and therefore eps_visc's discretisation -- fixed. The cost
# is that the physical solution changes down the sweep, so the comparison that means anything
# is ALWAYS first-order against higher-order AT THE SAME mu, never across mu.
CELLS=${CELLS:-"0.05:2:4" "0.02:2:4" "0.01:2:4" "0.005:2:4"}
QUICK=${QUICK:-1}

ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "limiter: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_limiter/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

# SFEM_LIMITER_STATS=1 so each arm also reports how often the reconstruction left the
# bound and how often the limiter acted. That is the other half of "less diffusive":
# a dissipation reduction is only attributable to the limiter if the limiter did
# something, and on the cavity at Re 100 it acts on half to four fifths of samples.
BASE="SFEM_CASE=step SFEM_U=1 SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=vanka SFEM_GMG_OMEGA=1.0 SFEM_LIMITER_STATS=1"

# A cap per ARM, set from the first-order run rather than guessed.
#
# The whole subject of this job is arms that do not converge, and an arm that does not
# converge does not fail quickly -- the bounded-face clip ran 399 Newton steps and 103,441
# linear iterations before giving up. With twenty arms in a 28-minute debug allocation, one
# of those consumes the wall clock and every arm after it is missing, which reads as a job
# that crashed rather than as one arm that stalled.
#
# The reference is the first-order run in the same cell, on the same mesh and hardware, so
# the cap travels with the problem size instead of being a number that has to be revisited.
# Eight times it is deliberately generous: the deferred correction is a fixed-point iteration
# on a lagged source and is EXPECTED to take more steps than first order, so the cap must
# only catch an arm that is stuck, never one that is merely slow. REF_SECS is set by the
# first run_one of each cell and the floor keeps the cap sane when that run is quick.
REF_SECS=0
arm_timeout() {
    local cap=$(( REF_SECS * 8 ))
    [ "$cap" -lt 120 ] && cap=120
    [ "$cap" -gt 900 ] && cap=900
    echo "$cap"
}

run_one() {
    local tag=$1; shift
    local log="$OUT/$tag.log"
    local t0=$SECONDS
    timeout "$(arm_timeout)" \
      stdbuf -oL -eL env $BASE "$@" SFEM_DIAG_CSV="$OUT/$tag.csv" \
        uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    local rc=$?
    # 124 is the cap firing. Reported as its own state: "stalled past the cap" and "diverged"
    # are different findings and collapsing them into one exit code loses the distinction the
    # job exists to draw.
    [ "$rc" -eq 124 ] && echo "TIMEOUT after $(arm_timeout)s" >> "$log" ndof conv nt li en rel
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

# CELLS is mu:macros:level per cell, overridable so this harness can sweep the CELL PECLET
# NUMBER instead of the limiter arms. That is worth asking separately: at the default cells
# the step runs at cell Peclet ~0.9-1.7, where donor-cell upwinding is barely active and
# removing its diffusion is worth 3-4%, while the manufactured solution at Peclet ~6 shows a
# 12x error reduction for the same schemes. "Less diffusive than upwind" is therefore a claim
# about Peclet, and this lets the dissipation and the accuracy be measured on ONE case.
#
# QUICK=1 keeps only what a diffusion comparison needs -- first order as the reference,
# unlimited frozen as the ceiling, bounded frozen as the scheme one would run. The unfrozen
# arms and the K sweep are already measured and would cost the time the Peclet axis wants.
if [ -n "${QUICK:-}" ]; then
    LM_ARMS=""; K_ARMS=""; FROZEN_ARMS="lim0:0:0 lim2:2:0"
else
    LM_ARMS="0 1 2 3"; K_ARMS="0.01 0.1 1 5 30"
    FROZEN_ARMS="lim0:0:0 lim1:1:0 lim2:2:0 lim2k5:2:5"
fi

HDR
for CELL in ${CELLS:-"0.05:2:4" "0.05:4:4"}; do
    MU=${CELL%%:*}; REST=${CELL#*:}; NN=${REST%%:*}; LV=${REST##*:}
    RE=$(python3 -c "print(int(1*1*2/$MU))")
    # First order, for the reference this is all measured against.
    REF_SECS=0
    REF_T0=$SECONDS
    run_one "re${RE}_n${NN}l${LV}_lo" SFEM_MU=$MU SFEM_N=$NN \
        SFEM_ELEMENT_REFINE_LEVEL=$LV SFEM_CONV_HO=0
    REF_SECS=$(( SECONDS - REF_T0 ))
    for LM in $LM_ARMS; do
        run_one "re${RE}_n${NN}l${LV}_ho_lim${LM}" SFEM_MU=$MU SFEM_N=$NN \
            SFEM_ELEMENT_REFINE_LEVEL=$LV SFEM_CONV_HO=1 SFEM_CONV_LIMITER=$LM
    done
    # Venkatakrishnan's eps^2 deactivation, swept over its severity.
    #
    # K is the knob that decides how much local variation counts as "smooth enough to leave
    # alone", so sweeping it is sweeping the severity of the approximation the arm makes --
    # and K = 0 is the zero-severity control, which is why lim2 above is run without it: that
    # run and K = 0 are the same arithmetic bit for bit (tests/cvfem_venkata_test.cpp proves
    # it over 512 input combinations), so the sweep starts from a point already in the table
    # rather than from a different scheme.
    #
    # The hypothesis being tested is specific. The stall is not a smoothness failure -- the
    # smooth arm stalls too -- it is the two-node bound collapsing as h -> 0 on flow-aligned
    # edges, where u_i and u_j agree to round-off and the clip is decided by noise. eps^2 is
    # the only mechanism in the scheme that can switch the limiter OFF there. If stall_Re
    # climbs with K towards the unlimited arm's reach, that is the diagnosis confirmed; if it
    # does not move, the bound needs replacing and limiter 3 is the arm that replaces it.
    #
    # The range spans four decades because the right K is not known a priori and is the
    # quantity being measured: Venkatakrishnan's own values are O(1) for a non-dimensional
    # solution, but this scheme's eps^2 is compared against velocity differences on a
    # flow-aligned edge, which are far smaller than the field itself.
    for K in $K_ARMS; do
        run_one "re${RE}_n${NN}l${LV}_ho_lim2_k${K}" SFEM_MU=$MU SFEM_N=$NN \
            SFEM_ELEMENT_REFINE_LEVEL=$LV SFEM_CONV_HO=1 SFEM_CONV_LIMITER=2 SFEM_VENKAT_K=$K
    done
    # Limiter freezing, which is the OTHER remedy and the interesting one.
    #
    # eps^2 buys convergence by relaxing the bound, and the transfer-function table in
    # tests/cvfem_venkata_test.cpp shows it relaxes it everywhere rather than only where the
    # field is smooth: at eps^2 = 1 the arm already retains more than the room available.
    # Freezing buys convergence a different way -- it removes the fixed point the lagged
    # source has to reach, by building the correction once per continuation stage and holding
    # it -- and costs nothing in boundedness, because the correction it holds is one the
    # unfrozen arm would also have produced.
    #
    # So the two are orthogonal and the question is which is needed. Frozen at K = 0 is the
    # decisive arm: if it converges like K = 5 while keeping K = 0's bound, freezing is the
    # remedy and eps^2 is a knob nobody has to spend. The clip is frozen too, because it is
    # the arm with the hardest active set and therefore the most to gain.
    for FA in $FROZEN_ARMS; do
        LBL=${FA%%:*}; REST=${FA#*:}; LM=${REST%%:*}; KK=${REST##*:}
        run_one "re${RE}_n${NN}l${LV}_ho_${LBL}_frozen" SFEM_MU=$MU SFEM_N=$NN \
            SFEM_ELEMENT_REFINE_LEVEL=$LV SFEM_CONV_HO=1 SFEM_CONV_LIMITER=$LM \
            SFEM_VENKAT_K=$KK SFEM_CONV_FREEZE=1
    done
    echo
done
echo "logs in $OUT"
echo "done $(date)"
