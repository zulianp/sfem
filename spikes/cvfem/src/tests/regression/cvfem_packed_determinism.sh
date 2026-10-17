#!/usr/bin/env bash
# Repeat each configuration to catch schedule-dependent changes.
# The separate fixed-input test varies thread count while keeping inputs and packing fixed.
# This gate previously caught nested parallel regions that made every thread sweep all packs.
set -u
BENCH=${1:?usage: $0 <cvfem_hex8_ns_upwind_bench>}
N=${N:-20}
THREADS=${THREADS:-}
if [ -z "$THREADS" ]; then
    # More than one, or the test is vacuous; and above the pack count there is nothing to race.
    THREADS=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 4 )
    [ "$THREADS" -gt 8 ] && THREADS=8
fi
[ "$THREADS" -lt 2 ] && { echo "only $THREADS cpu: nothing to race"; exit 77; }

FAIL=0
fp() {
    OMP_NUM_THREADS="$THREADS" "$BENCH" --n "$N" --repeat 2 --warmup 1 "$@" 2>/dev/null |
        sed -n 's/^ *fingerprint: *//p' | head -1
}

check() {
    desc="$1"; shift
    a=$(fp "$@"); b=$(fp "$@")
    if [ -z "$a" ]; then
        printf '%-54s FAIL (no fingerprint; configuration refused?)\n' "$desc"
        FAIL=$((FAIL + 1))
    elif [ "$a" = "$b" ]; then
        printf '%-54s OK   %s\n' "$desc" "$a"
    else
        printf '%-54s FAIL %s vs %s\n' "$desc" "$a" "$b"
        FAIL=$((FAIL + 1))
    fi
}

echo "== the packed layout reproduces itself bit for bit at $THREADS threads (n=$N)"
# One per operation and per geometry: the sweeps are separate functions now, so a nested region
# in one launcher says nothing about the others.
check "residual, affine"            --layout packed
check "residual, isoparam"          --layout packed --geom isoparam
check "residual + rhie-chow"        --layout packed --rhie-chow
check "residual + higher order"     --layout packed --rhie-chow --conv-ho 2
check "jacobian action, affine"     --layout packed --jac-action
check "jacobian action, isoparam"   --layout packed --jac-action --geom isoparam
check "assembly, affine"            --layout packed --assemble
check "assembly, isoparam"          --layout packed --assemble --geom isoparam
check "assembly, store layout"      --layout store --assemble
check "bsr apply"                   --layout packed --bsr-apply

# The ELEMENT-coloured layout is the other atomics-free one, and its barrier between colours is
# what makes it so; a colour loop hoisted into the wrong place would show here.
check "residual, element colouring" --layout ecolor

if [ "$FAIL" -ne 0 ]; then
    echo "cvfem_packed_determinism: $FAIL configuration(s) are not reproducible"
    exit 1
fi
echo "cvfem_packed_determinism: all reproducible"
