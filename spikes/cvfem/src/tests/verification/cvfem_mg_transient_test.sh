#!/usr/bin/env bash
# The transient multigrid/Vanka path, which nothing else exercises.
#
# Both existing driver-level tests pin SFEM_GMG=0 SFEM_PRECOND=direct, so every transient number
# this spike has ever gated came out of the dense coarse solve. The V-cycle carries the same
# operator and must reach the same answer: a preconditioner chooses how a linear system is solved,
# never what it solves, so a disagreement here is a defect and not a tolerance to be widened.
#
# The pump at N=4 and refine level 2, because it is the cheapest configuration that is BOTH
# semi-structured -- SFEM_GMG needs a refine level > 1, or there is no hierarchy to cycle on -- and
# carries the exact swept-volume identity. Measured while writing this: the direct arm 36 s against
# 1 s for multigrid, with both reporting u_linf 8.140670e+00 and p_linf 4.247488e+01. The 36x is
# the dense coarse factorisation being cubic in a problem that grew 8x under refinement; at N=8 the
# same arm costs 689 s, which is why this test is at N=4 and not larger.
set -u

DRIVER=${1:?usage: cvfem_mg_transient_test.sh <cvfem_hex8_ns_ssgmg>}
[ -x "$DRIVER" ] || { echo "mg_transient: no driver at $DRIVER" >&2; exit 1; }

WORK=$(mktemp -d 2>/dev/null || mktemp -d -t cvfem_mg_transient)
trap 'rm -rf "$WORK"' EXIT

# SFEM_FGMRES=1 explicitly. The driver defaults to BiCGStab whenever multigrid is off, and a
# multigrid cycle is not a fixed linear preconditioner -- the semi-structured restriction
# accumulates with atomics, so the cycle differs in its last bits between runs and BiCGStab's short
# recurrence has no right to converge. The direct arm sets it too, so the two differ in the
# preconditioner alone and not in the Krylov method.
COMMON="SFEM_CASE=pump SFEM_N=4 SFEM_MU=0.05 SFEM_U=1 SFEM_FGMRES=1 SFEM_NL_MAX_IT=20
        SFEM_DT=0.1 SFEM_NSTEPS=3 SFEM_PUMP_PERIOD=1 SFEM_BDF_ORDER=2 SFEM_ENABLE_OUTPUT=0
        SFEM_ELEMENT_REFINE_LEVEL=2"

fail=0
say() { printf '%-58s %s\n' "$1" "$2"; [ "$2" = OK ] || fail=1; }

env $COMMON SFEM_GMG=0 SFEM_PRECOND=direct "$DRIVER" "$WORK/d" > "$WORK/direct.log" 2>&1
rc_direct=$?
env $COMMON SFEM_GMG=1 "$DRIVER" "$WORK/m" > "$WORK/mg.log" 2>&1
rc_mg=$?

[ $rc_direct -eq 0 ] && say "the direct arm converges" OK || { say "the direct arm converges" FAIL; tail -5 "$WORK/direct.log"; }
[ $rc_mg -eq 0 ]     && say "the multigrid arm converges" OK || { say "the multigrid arm converges" FAIL; tail -5 "$WORK/mg.log"; }

# That the arms are what they claim to be. Without this the test passes just as happily when
# SFEM_GMG is silently ignored and both arms run the same preconditioner, which is the failure
# mode a preconditioner A/B is most exposed to.
grep -q "geometric multigrid" "$WORK/mg.log" \
    && say "the multigrid arm really used a V-cycle" OK \
    || say "the multigrid arm really used a V-cycle" FAIL
grep -q "geometric multigrid" "$WORK/direct.log" \
    && say "the direct arm did NOT use a V-cycle" FAIL \
    || say "the direct arm did NOT use a V-cycle" OK
grep -q "semi_structured: 1" "$WORK/mg.log" \
    && say "the mesh is semi-structured (a hierarchy exists)" OK \
    || say "the mesh is semi-structured (a hierarchy exists)" FAIL

# The answers themselves. Compared numerically to a relative 1e-6 rather than as strings: both arms
# drive Newton to SFEM_NL_RTOL 1e-8, so they agree to roughly that and not to the last bit, and a
# string comparison would fail on a one-ulp difference in a printed digit while a 1e-6 band is
# still four orders tighter than any real disagreement between two preconditioners.
cmp_field() {   # field-name, awk-index
    local name=$1 idx=$2
    local a b
    a=$(grep -E "^u_linf:" "$WORK/direct.log" | tail -1 | awk "{print \$$idx}")
    b=$(grep -E "^u_linf:" "$WORK/mg.log"     | tail -1 | awk "{print \$$idx}")
    if [ -z "$a" ] || [ -z "$b" ]; then
        say "$name is reported by both arms" FAIL
        return
    fi
    awk -v a="$a" -v b="$b" -v n="$name" 'BEGIN{
        d = (a > b ? a - b : b - a); s = (a < 0 ? -a : a); if (s == 0) s = 1;
        if (d / s < 1e-6) printf "%-58s %s\n", "multigrid and direct agree on " n, "OK";
        else { printf "%-58s %s\n", "multigrid and direct agree on " n, "FAIL";
               printf "    direct   : %s\n    multigrid: %s\n", a, b; exit 1 }
    }' || fail=1
}
cmp_field "u_linf" 2
cmp_field "p_linf" 4

# The physics identity, not just an A/B. The pump's port flux equals the volume its diaphragm
# swept, and that holds or does not hold independently of which preconditioner produced the state --
# so if the V-cycle has quietly changed the answer, this says so in units of the thing being
# conserved rather than in a norm.
#
# ASSERTED PER ARM AND RELATIVELY, not by comparing the two arms' output. The first version of this
# test compared the printed lines as strings and failed: direct 1.110223e-16 against multigrid
# 2.597310e-10. That was the assertion being wrong, not multigrid -- the same string-versus-numeric
# mistake this file argues against for u_linf twenty lines above, made again here.
#
# It was measured rather than assumed, because the alternative (multigrid genuinely leaking mass)
# would have meant the test was right and the code wrong. Sweeping the Newton tolerance:
#
#   multigrid  rtol 1e-8   |port - swept| 2.597310e-10
#   multigrid  rtol 1e-10  |port - swept| 2.364775e-14      <- four orders better
#   multigrid  rtol 1e-12  |port - swept| 2.364775e-14      <- saturates
#   direct     rtol 1e-8   |port - swept| 1.110223e-16
#
# So the gap is tolerance-limited: the arms stop at different points on Newton's quadratic tail
# (33 Newton iterations against 28), and mass is conserved in both. The threshold therefore comes
# from what a REAL leak looks like -- the historical rim bug sent 0.196 of a swept 1.000 out
# through the walls -- and not from the number that happened to be observed. 1e-6 relative sits
# four orders above the noise and five below that leak.
#
# Relative rather than absolute because `swept` = rho * U * Lx * Lz * pump_scale, so it tracks the
# case setup; a hardcoded absolute bound would silently become meaningless if U, rho or the domain
# ever changed.
identity() {   # label, logfile
    local lab=$1 log=$2 swept gap
    swept=$(grep -E "^pump: swept " "$log" | tail -1 | awk '{print $3}')
    gap=$(grep -F "pump: |port - swept|" "$log" | tail -1 | awk '{print $5}')
    if [ -z "$swept" ] || [ -z "$gap" ]; then
        say "$lab: the swept-volume identity is reported" FAIL
        return
    fi
    awk -v s="$swept" -v g="$gap" -v lab="$lab" 'BEGIN{
        a = (s < 0 ? -s : s); if (a == 0) a = 1;
        r = g / a;
        if (r < 1e-6)
            printf "%-58s %s\n", lab ": mass is conserved (rel " sprintf("%.2e", r) ")", "OK";
        else {
            printf "%-58s %s\n", lab ": mass is conserved", "FAIL";
            printf "    swept %s   |port - swept| %s   relative %.3e\n", s, g, r;
            exit 1
        }
    }' || fail=1
}
identity "direct" "$WORK/direct.log"
identity "multigrid" "$WORK/mg.log"

echo
[ $fail -eq 0 ] && echo PASSED || echo FAILED
exit $fail
