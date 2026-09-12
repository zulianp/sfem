#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-smooth0
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=smooth0_%j.out
#SBATCH --error=smooth0_%j.err
#
# Step 0 of the coarse-space work: the smoother, alone, with no coarse space at all.
#
# This runs first and nothing above it is allowed to be reported until it passes, because a
# broken smoother is indistinguishable from a broken coarse grid if you only look at the
# cycle. That is not a hypothetical here -- it already cost several rounds of coarse-operator
# investigation in this spike, and the fault turned out to be damping that made the smoother
# asymptotically divergent.
#
# It is re-run now because the Rhie-Chow time-scale fix changed the operator's spectrum, so
# every previously recorded smoother number is stale. Two questions, in order.
#
# QUESTION 1: does the stabilisation still dominate the operator?
#
# The block-split gate recorded, under the old coefficient, block norms of uu 0.284,
# up 2.231, pu 0.552 and pp 23.748 -- the pressure-pressure block about EIGHTY-FIVE times the
# momentum block. Two conclusions were drawn from that: that the system is not
# coupling-limited, and that a saddle-point smoother has nothing to work with, because
# SIMPLE's Schur complement reduces to Dpp within a couple of percent. Both conclusions rest
# on that ratio, and the corrected time scale is smaller by roughly the cell Peclet number,
# so the ratio is exactly what has to be re-measured before either conclusion is reused.
#
# QUESTION 2: is the smoother convergent, measured to its asymptote?
#
# SFEM_GMG_CHECK_IT defaults to 10 and that is not enough to answer it. The recorded failure
# read 0.88 rising to 0.95 over the first eight sweeps, looked convergent, and was still
# rising through 1.038 at sweep 39. So this runs 150 sweeps, as the measurement that caught it
# did, and reports the rate over the LAST five rather than the first few. A rate still moving
# at sweep 149 means the measurement is not finished, which the report says explicitly.
#
# A rate near 0.97 is not by itself a bad smoother: the asymptotic rate is set by the
# smoothest mode, which is what the coarse grid exists to remove, and good multigrid
# smoothers routinely look poor measured this way. What is fatal is a rate at or above 1.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "smooth0: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_smooth0/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

# The semi-structured configuration the recorded numbers were taken on, so the comparison is
# against them and not against a different problem.
BASE="SFEM_CASE=poiseuille SFEM_LX=4 SFEM_LY=1 SFEM_MU=0.01 SFEM_U=1 SFEM_N=2
      SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_GMG=1 SFEM_NL_MAX_IT=8"

echo "== question 1: the block norms, old time scale against new"
echo "   INITIAL STATE ONLY, and that limits what they can settle. check_transfers runs once"
echo "   at setup, before the Newton loop, so there is no iteration to ask it for -- and at"
echo "   setup the interior velocity is zero, only the Dirichlet values are set. The"
echo "   advective branch of the time scale is therefore absent by construction here, and"
echo "   these norms report the diffusive limit whatever the flow is: pp moves by exactly the"
echo "   factor two between the two coefficients, which is h^2/(2 nu) against h^2/(4 nu) and"
echo "   nothing to do with the Peclet number. Re-evaluating the recorded 85x ratio needs the"
echo "   check moved inside the Newton loop; that is a follow-up, not smuggled in here."
printf '%-10s %10s %10s %10s %10s %10s\n' tau uu up pu pp pp/uu
for T in 0 1; do
    log="$OUT/blocks_tau$T.log"
    stdbuf -oL -eL env $BASE SFEM_RC_TAU=$T SFEM_GMG_CHECK=1 \
        uenv run $U -- "$D" "$OUT/blk$T" > "$log" 2>&1
    # the gate prints the four norms on one line; take the first occurrence
    ln=$(grep -E "^block split:" "$log" | head -1)
    if [ -z "$ln" ]; then printf '%-10s %s\n' "tau=$T" "(no block split line -- see $log)"; continue; fi
    set -- $(echo "$ln" | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?' | head -4)
    uu=${1:-0}; up=${2:-0}; pu=${3:-0}; pp=${4:-0}
    r=$(awk -v a="$pp" -v b="$uu" 'BEGIN{ if (b+0==0) print "-"; else printf "%.1f", a/b }')
    printf '%-10s %10s %10s %10s %10s %10s\n' "tau=$T" "$uu" "$up" "$pu" "$pp" "$r"
done
echo

echo "== question 2: standalone smoother rate at 150 sweeps, rate over sweeps 145-149"
echo "   At TWO Newton iterations. SFEM_GMG_CHECK_AT=0 is where this gate has always fired"
echo "   and has a zero interior velocity; =2 is a developed field. They are different"
echo "   operators and they give different rates -- block-Jacobi at omega 0.35 measured"
echo "   0.9684, 0.9584 and 0.9777 at iterations 0, 1 and 2 -- so the developed one is the"
echo "   number that describes the solve."
echo "   VERDICT is on the asymptote: >=1 is fatal, and 'still-moving' means the"
echo "   measurement did not finish and the number below it must not be used."
printf '%-12s %5s %4s %3s %10s %10s %-14s\n' smoother omega tau at ndof rate verdict
for AT in 0 2; do
 for T in 0 1; do
  for S in bjacobi vanka simple; do
    for OM in 0.35 0.7 1.0; do
        tag="${S}_om${OM}_tau${T}_at${AT}"
        log="$OUT/$tag.log"
        # Both omega variables: the standalone probe reads SFEM_VANKA_OMEGA for vanka and
        # SFEM_GMG_OMEGA for bjacobi and simple, and setting only one silently measures a
        # different damping than the row claims -- which is the exact mistake that let a
        # divergent smoother be certified once already.
        stdbuf -oL -eL env $BASE SFEM_RC_TAU=$T SFEM_SMOOTHER=$S \
            SFEM_VANKA_OMEGA=$OM SFEM_GMG_OMEGA=$OM \
            SFEM_GMG_CHECK=3 SFEM_GMG_CHECK_IT=150 SFEM_GMG_CHECK_AT=$AT \
            uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
        ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
        # the asymptote, not the first sweeps: mean rate over the last five reported
        rate=$(grep -E "^  sweep" "$log" | tail -5 | awk '{s+=$NF; n++} END{ if(n) printf "%.4f", s/n }')
        # and whether it is still moving -- last five against the five before them
        prev=$(grep -E "^  sweep" "$log" | tail -10 | head -5 | awk '{s+=$NF; n++} END{ if(n) printf "%.4f", s/n }')
        v=$(awk -v r="$rate" -v p="$prev" 'BEGIN{
                if (r=="") { print "no-rate"; exit }
                if (r+0 >= 1) { print "DIVERGENT"; exit }
                d = r-p; if (d<0) d=-d;
                if (d > 0.002) print "still-moving"; else print "convergent" }')
        if [ -z "$rate" ]; then v="gate-did-not-fire"; fi
        printf '%-12s %5s %4s %3s %10s %10s %-14s\n' "$S" "$OM" "$T" "$AT" "${ndof:--}" "${rate:--}" "$v"
    done
  done
 done
done
echo
echo "logs in $OUT"
echo "done $(date)"
