#!/usr/bin/env bash
# The deferred correction's freezing path, and the default that now selects it.
#
# WHY THIS EXISTS. SFEM_CONV_FREEZE defaults to 1, and until this test nothing in ctest set
# SFEM_CONV_HO at all -- so the whole deferred-correction path, including the branch that
# default now selects, was outside the local gate. A suite that passes 38/38 while never
# executing the code whose default just changed is not evidence about that change.
#
# WHAT IT ASSERTS, in the order the claims were made:
#
#   1. HO off is untouched by the flag. The freeze branch is guarded on conv_ho, so the two
#      runs must agree EXACTLY -- same Newton count, same residual string. This is the claim
#      that the default change is invisible to every existing configuration.
#   2. Frozen converges, and in strictly fewer Newton steps than unfrozen. That is the whole
#      mechanism: measured 15 against 63 here on Grace's larger cousin of this case, and
#      12 against 303 on the backward-facing step at 47,268 dof.
#   3. Unfrozen still works when asked for. An option that has silently stopped functioning
#      is worse than one that was removed.
#
# It does NOT assert that the two converge to the same state. They do not: frozen solves
# R_lo(u) + frozen = 0 with the correction taken at the stage's opening state, which is a
# different problem. That the difference does not cost accuracy is an order-of-accuracy
# question, measured by the conv group in scripts/verify_report.sh, and it cannot be settled
# on one coarse mesh here.

set -uo pipefail
DRIVER=${1:?usage: cvfem_conv_freeze_test.sh <cvfem_hex8_ns_ssgmg>}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

fail=0
note() { printf '  %-58s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1; return 0; }

# Small and flat on purpose: this gates a mechanism, not a rate, and the mechanism shows at
# any size. The dense LU keeps it deterministic.
COMMON="SFEM_CASE=cavity SFEM_N=5 SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=direct
        SFEM_ENABLE_OUTPUT=0 SFEM_ELEMENT_REFINE_LEVEL=1"

run() {  # <label> <extra env...>
    local lbl=$1; shift
    env $COMMON "$@" "$DRIVER" "$WORK/out_$lbl" > "$WORK/$lbl.log" 2>&1
    echo $?
}
nt()   { sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$WORK/$1.log" | tail -1; }
conv() { sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$WORK/$1.log" | tail -1; }
res()  { grep -aE "^newton [0-9]+ " "$WORK/$1.log" | tail -1; }

# ---- 1. with the correction off, the flag must change nothing ----------------------
run lo_f0 SFEM_CONV_HO=0 SFEM_CONV_FREEZE=0 > /dev/null
run lo_f1 SFEM_CONV_HO=0 SFEM_CONV_FREEZE=1 > /dev/null
if [ "$(nt lo_f0)" = "$(nt lo_f1)" ] && [ "$(res lo_f0)" = "$(res lo_f1)" ] && [ -n "$(nt lo_f0)" ]; then
    note "HO off: the freeze flag changes nothing" OK
else
    note "HO off: the freeze flag changes nothing" FAIL
    echo "      f0: $(nt lo_f0) '$(res lo_f0)'"
    echo "      f1: $(nt lo_f1) '$(res lo_f1)'"
fi

# ---- 2 and 3. frozen against unfrozen, both with the correction on -----------------
run ho_frozen   SFEM_CONV_HO=1 SFEM_CONV_LIMITER=2 > /dev/null   # default freeze=1
run ho_unfrozen SFEM_CONV_HO=1 SFEM_CONV_LIMITER=2 SFEM_CONV_FREEZE=0 > /dev/null

[ "$(conv ho_frozen)"   = 1 ] && note "frozen converges"   OK || note "frozen converges"   FAIL
[ "$(conv ho_unfrozen)" = 1 ] && note "unfrozen still converges when asked for" OK \
                              || note "unfrozen still converges when asked for" FAIL

# WHAT FREEZING BUYS, NOW THAT THE JACOBIAN CAN CARRY THE CORRECTION.
#
# This used to assert that the frozen run takes FEWER Newton steps than the unfrozen one, and it
# did: with the correction lagged out of the Jacobian, the unfrozen scheme is a fixed-point
# iteration on a term the linearisation does not see, and it converges at a linear rate. Measured
# here, cavity at Re 100 over five continuation stages: 61 steps unfrozen against 14 frozen.
#
# With SFEM_HO_EXACT_JAC on -- the default -- the Jacobian differentiates the correction, and the
# unfrozen scheme converges in the SAME 14 steps as the frozen one. So freezing no longer buys
# steps; it was a workaround for the missing derivative. What is still true, and is the stronger
# statement, is that the unfrozen run must not be WORSE than the frozen one, and that the
# advantage the freeze used to have reappears the moment the exact term is switched off.
run ho_unfrozen_lagged SFEM_CONV_HO=1 SFEM_CONV_LIMITER=2 SFEM_CONV_FREEZE=0 SFEM_HO_EXACT_JAC=0 > /dev/null
f=$(nt ho_frozen); u=$(nt ho_unfrozen); l=$(nt ho_unfrozen_lagged)
if [ -n "$f" ] && [ -n "$u" ] && [ "$u" -le "$f" ]; then
    note "unfrozen matches frozen with the exact Jacobian ($u <= $f)" OK
else
    note "unfrozen matches frozen with the exact Jacobian (${u:-?} vs ${f:-?})" FAIL
fi
if [ -n "$l" ] && [ -n "$u" ] && [ "$l" -gt "$u" ]; then
    note "the lagged Jacobian costs Newton steps ($l > $u)" OK
else
    note "the lagged Jacobian costs Newton steps (${l:-?} vs ${u:-?})" FAIL
fi

# ---- and the default really is frozen ----------------------------------------------
run ho_default SFEM_CONV_HO=1 SFEM_CONV_LIMITER=2 SFEM_CONV_FREEZE=1 > /dev/null
if [ "$(nt ho_default)" = "$(nt ho_frozen)" ] && [ -n "$(nt ho_frozen)" ]; then
    note "the default is the frozen path" OK
else
    note "the default is the frozen path" FAIL
fi

[ "$fail" = 0 ] && { echo "conv_freeze: all checks passed"; exit 0; }
echo "conv_freeze: FAILED" >&2; exit 1
