#!/usr/bin/env bash
# The temporal order of the scheme, measured end to end through the full driver.
#
# tests/cvfem_ss_transient_test.cpp already proves the BDF TERM is first- and second-order, by
# differencing it against du/dt on a manufactured history. What a term-level test cannot see is
# everything the term is wrapped in: the Newton loop, FGMRES, the preconditioner, the time-varying
# boundary condition and the pressure gauge. Any one of those can cap the order while the term
# itself stays perfect -- a boundary condition evaluated at t_n instead of t_{n+1} caps everything
# at first order and no term test would notice. This measures what the driver actually delivers.
#
# No exact solution is needed, which is why this is Richardson and not an error study: three solves
# to the SAME final time at dt, dt/2 and dt/4 give
#     p = log2( |u_dt - u_dt/2| / |u_dt/2 - u_dt/4| )
# and the order falls out without a reference to compare against.
#
# VELOCITY ONLY. The pressure carries a gauge, and including it contaminates the fit -- measured,
# BDF1 reads 1.075 on velocity against 1.216 on the full state.
#
# WHY THE TWO SCHEMES USE DIFFERENT LADDERS, which is the part worth reading before changing
# anything here. A Richardson fit means nothing outside the asymptotic regime, and the two schemes
# enter it at different step sizes. Measured on this case while writing this test:
#
#   dt 0.1    /  4,8,16 steps   BDF2  ratio 1.16   order 0.211   far outside; the fit is noise
#   dt 0.025  /  8,16,32 steps  BDF2  ratio 3.980  order 1.993   in regime
#   dt 0.025  /  8,16,32 steps  BDF1  ratio 1.661  order 0.732   BDF1 NOT yet in regime here
#   dt 0.025  / 16,32,64 steps  BDF1  ratio 2.060  order 1.043   in regime
#
# The 0.211 above is why this test exists in this shape: it looked exactly like a broken
# second-order scheme and was entirely an artefact of too coarse a dt. So BDF2 is gated on the
# shorter ladder and BDF1 on the longer one. That is not a convenience -- fitting BDF1 on the short
# ladder reports 0.73 for a scheme that is genuinely first order, and the only ways to make that
# pass are to widen the band until it gates nothing, or to run longer. This runs longer.
#
# THE RATIO IS ASSERTED AS WELL AS THE ORDER. The ratio is what says the fit was in regime at all;
# an order can land inside its band by accident while the differences are not converging, which is
# precisely how the 0.211 case first presented.
set -u

DRIVER=${1:?usage: cvfem_richardson_test.sh <cvfem_hex8_ns_ssgmg> <python>}
PYTHON=${2:?usage: cvfem_richardson_test.sh <cvfem_hex8_ns_ssgmg> <python>}
[ -x "$DRIVER" ] || { echo "richardson: no driver at $DRIVER" >&2; exit 1; }
"$PYTHON" -c "import numpy" >/dev/null 2>&1 \
    || { echo "richardson: $PYTHON has no numpy" >&2; exit 1; }

WORK=$(mktemp -d 2>/dev/null || mktemp -d -t cvfem_richardson)
trap 'rm -rf "$WORK"' EXIT

# SFEM_FGMRES=1 explicitly: the driver defaults to BiCGStab whenever multigrid is off, and a
# convergence study run on a Krylov method that breaks down unpredictably measures the breakdown.
# The direct preconditioner keeps the linear solve out of the result entirely.
BASE="SFEM_CASE=pump SFEM_N=4 SFEM_MU=0.05 SFEM_U=1 SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=direct
      SFEM_NL_MAX_IT=20 SFEM_PUMP_PERIOD=1 SFEM_ENABLE_OUTPUT=0"

fail=0
say() { printf '%-58s %s\n' "$1" "$2"; [ "$2" = OK ] || fail=1; }

solve() {   # order dt steps tag
    env $BASE SFEM_BDF_ORDER=$1 SFEM_DT=$2 SFEM_NSTEPS=$3 SFEM_RESTART_OUT="$WORK/$4" \
        "$DRIVER" "$WORK/out_$4" > "$WORK/$4.log" 2>&1
    local rc=$?
    rm -rf "$WORK/out_$4"
    if [ $rc -ne 0 ]; then
        say "BDF$1 solve at dt=$2 converges" FAIL
        tail -4 "$WORK/$4.log"
        return 1
    fi
    [ -f "$WORK/$4/state.float64" ] || { say "BDF$1 dt=$2 wrote a state" FAIL; return 1; }
    return 0
}

# order tag_a tag_b tag_c order_lo order_hi ratio_lo ratio_hi
ladder() {
    local ord=$1 a=$2 b=$3 c=$4 olo=$5 ohi=$6 rlo=$7 rhi=$8
    local out
    out=$("$PYTHON" - "$WORK/$a/state.float64" "$WORK/$b/state.float64" "$WORK/$c/state.float64" <<'PYEOF'
import sys, numpy as np
a, b, c = (np.fromfile(p, dtype=np.float64) for p in sys.argv[1:4])
if not (a.size == b.size == c.size):
    print("SIZE_MISMATCH", a.size, b.size, c.size); raise SystemExit(0)
mask = np.ones(a.size, dtype=bool)
mask[3::4] = False                      # AoS [ux,uy,uz,p]: drop the gauged pressure
d1 = np.linalg.norm(a[mask] - b[mask])
d2 = np.linalg.norm(b[mask] - c[mask])
if d2 <= 0:
    print("ZERO_DIFFERENCE", d1, d2); raise SystemExit(0)
print(f"{d1:.6e} {d2:.6e} {d1/d2:.4f} {np.log(d1/d2)/np.log(2.0):.4f}")
PYEOF
)
    case "$out" in
        SIZE_MISMATCH*|ZERO_DIFFERENCE*|"")
            say "BDF$ord Richardson fit is computable" FAIL
            printf '    %s\n' "$out"
            return ;;
    esac
    local d1 d2 ratio order
    read -r d1 d2 ratio order <<EOF
$out
EOF
    printf '  BDF%d  d1=%s  d2=%s  ratio=%s  observed order=%s\n' "$ord" "$d1" "$d2" "$ratio" "$order"
    awk -v r="$ratio" -v lo="$rlo" -v hi="$rhi" 'BEGIN{exit !(r>=lo && r<=hi)}' \
        && say "BDF$ord differences converge (ratio in [$rlo,$rhi])" OK \
        || say "BDF$ord differences converge (ratio in [$rlo,$rhi])" FAIL
    awk -v p="$order" -v lo="$olo" -v hi="$ohi" 'BEGIN{exit !(p>=lo && p<=hi)}' \
        && say "BDF$ord observed order is in [$olo,$ohi]" OK \
        || say "BDF$ord observed order is in [$olo,$ohi]" FAIL
}

# BDF2 on the short ladder: T = 0.2, and it is in regime there (ratio 3.980, order 1.993).
if solve 2 0.025 8 b2a && solve 2 0.0125 16 b2b && solve 2 0.00625 32 b2c; then
    ladder 2 b2a b2b b2c 1.80 2.20 3.40 4.60
fi

# BDF1 on the long ladder: T = 0.4. The short ladder reports 0.732 for it, which is the fit not
# being asymptotic rather than the scheme being wrong.
if solve 1 0.025 16 b1a && solve 1 0.0125 32 b1b && solve 1 0.00625 64 b1c; then
    ladder 1 b1a b1b b1c 0.90 1.20 1.80 2.30
fi

echo
[ $fail -eq 0 ] && echo PASSED || echo FAILED
exit $fail
