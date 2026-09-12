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
# This runs first and nothing above it is reported until it passes, because a broken smoother
# is indistinguishable from a broken coarse grid if you only look at the cycle. That already
# cost several rounds of coarse-operator investigation in this spike, and the fault turned out
# to be damping that made the smoother asymptotically divergent.
#
# It is re-run now because the Rhie-Chow time-scale fix changed the operator's spectrum, so
# every previously recorded smoother number is stale.
#
# FOUR THINGS THIS JOB GETS RIGHT, each because getting it wrong was tried first.
#
# 1. The gate iteration. Every SFEM_GMG_CHECK gate used to fire only at newton_it == 0, where
#    the interior velocity is ZERO -- only the Dirichlet values are set. The time scale now
#    carries |u|, so its advective branch is absent there by construction and the gate reports
#    the diffusive limit whatever the flow is. SFEM_GMG_CHECK_AT picks the iteration; this
#    measures 0 and 2, and they are different operators: block-Jacobi at omega 0.35 gave
#    0.9684, 0.9584 and 0.9777 at iterations 0, 1 and 2.
#
# 2. Both damping variables. The probe reads SFEM_VANKA_OMEGA for vanka and SFEM_GMG_OMEGA for
#    bjacobi and simple. Setting one is how a divergent smoother came to be certified here
#    once already -- the check reported 0.35 while the cycle ran 1.
#
# 3. Enough sweeps to see an asymptote. SFEM_GMG_CHECK_IT defaults to 10. The failure this
#    rule exists for read 0.88 rising to 0.95 over eight sweeps and was still rising through
#    1.038 at sweep 39. Measured here, the rate needs about 300 sweeps to settle, so the
#    matrix runs 150 and anything unsettled is re-run at 600.
#
# 4. The right statistic. The probe prints `rate` as |r_n|/|r_{n-1}|, a SINGLE-SWEEP ratio,
#    and for a damped stationary iteration on a random error that is non-monotone and pokes
#    above 1 without diverging: the new time scale's worst single sweep is 1.0576 while its
#    geometric mean over the same 600 sweeps is 0.9770. A maximum single-sweep ratio is
#    therefore NOT a divergence criterion. The geometric mean over a window is.
#
# A rate near 0.97 is not by itself a bad smoother: the asymptotic rate is set by the
# smoothest mode, which is exactly what the coarse grid exists to remove, and good multigrid
# smoothers routinely look poor measured this way. What is fatal is a geometric mean at or
# above 1.
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

# Geometric-mean rates over the last third and the last sixth, from the residual norms. A gap
# between the two is a rate still drifting; one average alone hides it.
rates_of() {  # rates_of <log>  ->  "n geo_third geo_sixth first last"
    awk '/^  sweep/ {n++; r[n]=$4}
         END{ if (n < 20) { print "0 - - - -"; exit }
              t = int(n/3); x = int(n/6);
              printf "%d %.4f %.4f %.3e %.3e\n", n, (r[n]/r[n-t])^(1.0/t), (r[n]/r[n-x])^(1.0/x), r[1], r[n] }' "$1"
}

verdict_of() {  # verdict_of <geo_third> <geo_sixth>
    awk -v a="$1" -v b="$2" 'BEGIN{
        if (b+0 >= 1 || a+0 >= 1) { print "DIVERGENT"; exit }
        d = b-a; if (d<0) d=-d;
        if (d > 0.005) print "drifting"; else print "settled" }'
}

# The semi-structured configuration the recorded numbers were taken on, so the comparison is
# against them and not against a different problem. NL_MAX_IT has to allow Newton to reach the
# gate iteration.
BASE="SFEM_CASE=poiseuille SFEM_LX=4 SFEM_LY=1 SFEM_MU=0.01 SFEM_U=1 SFEM_N=1
      SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_GMG=1 SFEM_NL_MAX_IT=8"

echo "== question 1: the block norms, old time scale against new"
echo "   INITIAL STATE ONLY, and that bounds what they settle. check_transfers runs once at"
echo "   setup, before the Newton loop, so there is no iteration to ask it for -- and at setup"
echo "   the interior velocity is zero. The advective branch is absent here, so these norms"
echo "   report the diffusive limit: pp moves by exactly two between the coefficients, which"
echo "   is h^2/(2 nu) against h^2/(4 nu) and nothing to do with the Peclet number. The"
echo "   recorded 'pp is 85x uu, so a saddle-point smoother has nothing to work with' cannot"
echo "   be re-evaluated from this; that needs the check moved into the loop, as a follow-up."
printf '%-8s %12s %12s %12s %12s %8s\n' tau uu up pu pp pp/uu
for T in 0 1; do
    log="$OUT/blocks_tau$T.log"
    stdbuf -oL -eL env $BASE SFEM_RC_TAU=$T SFEM_GMG_CHECK=1 \
        uenv run $U -- "$D" "$OUT/blk$T" > "$log" 2>&1
    ln=$(grep -E "^block split:" "$log" | head -1)
    if [ -z "$ln" ]; then printf '%-8s %s\n' "tau=$T" "(no block split line -- see $log)"; continue; fi
    set -- $(echo "$ln" | grep -oE '[0-9]+\.[0-9]+e[-+][0-9]+' | head -4)
    r=$(awk -v a="${4:-0}" -v b="${1:-0}" 'BEGIN{ if (b+0==0) print "-"; else printf "%.1f", a/b }')
    printf '%-8s %12s %12s %12s %12s %8s\n' "tau=$T" "${1:--}" "${2:--}" "${3:--}" "${4:--}" "$r"
done
echo

echo "== question 2: the gate -- standalone smoother, 150 sweeps, geometric-mean rate"
printf '%-9s %5s %4s %3s %9s %8s %8s %10s %-13s\n' \
    smoother omega tau at ndof geo1/3 geo1/6 final_r verdict
: > "$OUT/matrix.txt"
for AT in 0 2; do
 for T in 0 1; do
  for S in bjacobi vanka simple; do
    for OM in 0.35 0.7 1.0; do
        tag="${S}_om${OM}_tau${T}_at${AT}"
        log="$OUT/$tag.log"
        stdbuf -oL -eL env $BASE SFEM_RC_TAU=$T SFEM_SMOOTHER=$S \
            SFEM_VANKA_OMEGA=$OM SFEM_GMG_OMEGA=$OM \
            SFEM_GMG_CHECK=3 SFEM_GMG_CHECK_IT=150 SFEM_GMG_CHECK_AT=$AT \
            uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
        ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
        set -- $(rates_of "$log")
        n=$1; g3=$2; g6=$3; rn=$5
        if [ "$n" = "0" ]; then
            v="gate-did-not-fire"; g3=-; g6=-; rn=-
        else
            v=$(verdict_of "$g3" "$g6")
        fi
        printf '%-9s %5s %4s %3s %9s %8s %8s %10s %-13s\n' \
            "$S" "$OM" "$T" "$AT" "${ndof:--}" "$g3" "$g6" "$rn" "$v" | tee -a "$OUT/matrix.txt"
    done
  done
 done
done
echo

# Anything unsettled at 150 sweeps has not shown its asymptote, and whether that asymptote is
# below 1 is the question step 0 exists to answer. Capped at six rows, which is what bounds the
# cost of a 28-minute allocation the matrix above has already spent most of.
DEEP=$(awk '$NF=="drifting" || $NF=="DIVERGENT" {print $1"_om"$2"_tau"$3"_at"$4}' "$OUT/matrix.txt" | head -6)
if [ -z "$DEEP" ]; then
    echo "== deep pass: nothing was drifting or divergent at 150 sweeps, so none needed"
else
    echo "== deep pass: 600 sweeps on the rows that had not settled"
    printf '%-26s %8s %8s %10s %-14s\n' tag geo1/3 geo1/6 final_r verdict
    for tag in $DEEP; do
        S=${tag%%_om*}
        OM=${tag#*_om}; OM=${OM%%_tau*}
        T=${tag#*_tau}; T=${T%%_at*}
        AT=${tag##*_at}
        log="$OUT/deep_$tag.log"
        stdbuf -oL -eL env $BASE SFEM_RC_TAU=$T SFEM_SMOOTHER=$S \
            SFEM_VANKA_OMEGA=$OM SFEM_GMG_OMEGA=$OM \
            SFEM_GMG_CHECK=3 SFEM_GMG_CHECK_IT=600 SFEM_GMG_CHECK_AT=$AT \
            uenv run $U -- "$D" "$OUT/deep_$tag" > "$log" 2>&1
        set -- $(rates_of "$log")
        n=$1; g3=$2; g6=$3; rn=$5
        if [ "$n" = "0" ] || [ "$n" -lt 100 ]; then
            printf '%-26s %8s %8s %10s %-14s\n' "$tag" - - - "incomplete"
        else
            printf '%-26s %8s %8s %10s %-14s\n' "$tag" "$g3" "$g6" "$rn" "$(verdict_of "$g3" "$g6")"
        fi
    done
fi
echo
echo "logs in $OUT"
echo "done $(date)"
