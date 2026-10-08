#!/bin/sh
# THE f32 BENCH AGAINST THE f64 ONE: every operator, every layout, entry by entry.
#
# cvfem_hex8_ns_upwind_bench_f32 is the same driver compiled with the computation type set to
# float, so every field, staged pack, cached geometry table and kernel runs in single precision.
# Its own --verify oracles are double-precision checks and it refuses them. This is its oracle:
# both builds run the same configuration with --dump, and the f32 output is held to the f64 one --
# the operator a finite difference has already verified (cvfem_bench_staging.sh).
#
# THE STATE IS PERTURBED (--perturb 1e-2, keyed on the node coordinates, identical in both builds).
# The timing state is linear in the coordinates, so many reconstructions land exactly on a limiter
# bound and some mass fluxes are exactly zero: kinks where the Jacobian takes a subgradient decided
# by the last bit, and the two builds' last bits differ by design. Unperturbed, the first-order
# Jacobian disagreed on 0.45% of dofs at 7e-3 and the clip on 34%; perturbed, both agree to a few
# ulp of float except as below. 1e-2 rather than the finite-difference check's 1e-3 because the
# limiter derivatives' band is 64 rounding units of each build's own type -- 8e-6 of the field in
# float, 1.4e-14 in double -- and a perturbation of 1e-3 left enough reconstructions within 8e-6 of
# a bound that the clip differed on 1.3% of dofs, every one of them the band doing its job.
#
# TWO STANDARDS, as the finite-difference check has:
#   smooth arms   (residuals, the first-order and unlimited Jacobians, the lagged one):
#                 max |f32 - f64| / max |f64| < 2e-5 -- about 170 ulp of float; measured
#                 3e-7 to 7e-7, and 2.8e-6 with the boundary closure and transient term.
#   limited arms  (the clip, Venkatakrishnan and Darwish-Moukalled Jacobians): fewer than 1% of
#                 dofs past 1e-4. The limiter derivatives take the interior subgradient inside a
#                 band of 64 ulp of the computation type (CVFEM_LIMITER_BAND_ULPS), so a
#                 reconstruction within that many float-ulp of a bound takes it in f32 and not in
#                 f64; measured 0.15% to 0.36% for the clip, under 0.1% for the other two.
# Both bounds sit orders of magnitude below what a field read at the wrong width or a lane
# stride bound to the other build's vector width produces, which is O(1) on most entries.
#
# Packed is NOT compared against atomic here: packing renumbers the mesh in place, so dumps from
# two processes are not index-comparable. Each layout is compared with itself across precisions,
# and the driver's own always-on check holds the f32 layouts to each other in one process.
set -u
B64="${1:?usage: cvfem_bench_f32.sh <bench f64> <bench f32> <python>}"
B32="${2:?usage: cvfem_bench_f32.sh <bench f64> <bench f32> <python>}"
PY="${3:-python3}"
N=${N:-10}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
FAIL=0

cat > "$WORK/cmp.py" <<'PYEOF'
import sys, array
def load(p):
    a = array.array('d')
    with open(p, 'rb') as f: a.frombytes(f.read())
    return a
a, b = load(sys.argv[1]), load(sys.argv[2])
kind = sys.argv[3]
if len(a) != len(b) or len(a) == 0:
    print("FAIL sizes %d %d" % (len(a), len(b))); sys.exit(1)
scale = max(abs(x) for x in a) or 1.0
d = [abs(x - y) / scale for x, y in zip(a, b)]
rmax = max(d)
frac = sum(1 for v in d if v > 1e-4) / len(d)
if kind == "smooth":
    ok = rmax < 2e-5
else:
    ok = frac < 0.01
print("%s rel_max %.3e  past_1e-4 %.4f%%" % ("OK  " if ok else "FAIL", rmax, 100.0 * frac))
sys.exit(0 if ok else 1)
PYEOF

run() {  # <kind> <key> <layout> <extra...>
    kind=$1; key=$2; lay=$3; shift 3
    if ! "$B64" --n "$N" --repeat 1 --warmup 0 --layout "$lay" --perturb 1e-2 "$@" \
            --dump "$WORK/a.bin" > "$WORK/a.log" 2>&1; then
        printf '%-34s %-7s FAIL (f64 run)\n' "$key" "$lay"; tail -3 "$WORK/a.log" | sed 's/^/    /'
        FAIL=$((FAIL + 1)); return
    fi
    if ! "$B32" --n "$N" --repeat 1 --warmup 0 --layout "$lay" --perturb 1e-2 "$@" \
            --dump "$WORK/b.bin" > "$WORK/b.log" 2>&1; then
        printf '%-34s %-7s FAIL (f32 run)\n' "$key" "$lay"; tail -3 "$WORK/b.log" | sed 's/^/    /'
        FAIL=$((FAIL + 1)); return
    fi
    # The f32 binary must say it computed in f32: a build that silently kept double would pass
    # every comparison below at 0.
    if ! grep -q "^  scalar: f32" "$WORK/b.log" || ! grep -q "^  scalar: f64" "$WORK/a.log"; then
        printf '%-34s %-7s FAIL (precision not as built)\n' "$key" "$lay"
        FAIL=$((FAIL + 1)); return
    fi
    if out=$("$PY" -I "$WORK/cmp.py" "$WORK/a.bin" "$WORK/b.bin" "$kind"); then
        printf '%-34s %-7s %s\n' "$key" "$lay" "$out"
    else
        printf '%-34s %-7s %s\n' "$key" "$lay" "$out"
        FAIL=$((FAIL + 1))
    fi
}

# Every layout the paper reports an f32 rate for, on every operator it reports: the element-coloured
# one carries the whole list; the pack-coloured one has no higher-order correction (the driver
# refuses it) and runs the first-order operators and the lagged action below.
for lay in packed atomic ecolor; do
    run smooth  "residual"                          $lay
    run smooth  "residual + Rhie-Chow"              $lay --rhie-chow
    run smooth  "residual + RC, boundary, transient" $lay --rhie-chow --boundary --transient 0.01
    run smooth  "residual + ho, unlimited"          $lay --rhie-chow --conv-ho 0
    run smooth  "residual + ho, clip"               $lay --rhie-chow --conv-ho 1
    run smooth  "residual + ho, Venkatakrishnan"    $lay --rhie-chow --conv-ho 2
    run smooth  "residual + ho, Darwish-Moukalled"  $lay --rhie-chow --conv-ho 3
    run smooth  "Jex bare"                          $lay --jac-action
    run smooth  "Jex + Rhie-Chow"                   $lay --jac-action --rhie-chow
    run smooth  "Jhoex unlimited"                   $lay --jac-action --rhie-chow --conv-ho 0
    run limited "Jhoex clip"                        $lay --jac-action --rhie-chow --conv-ho 1
    run limited "Jhoex Venkatakrishnan"             $lay --jac-action --rhie-chow --conv-ho 2
    run limited "Jhoex Darwish-Moukalled"           $lay --jac-action --rhie-chow --conv-ho 3
    run smooth  "Jlag (Venkatakrishnan lagged)"     $lay --jac-action --rhie-chow --conv-ho 2 --lagged-ho
done
run smooth  "residual"                          colored
run smooth  "residual + Rhie-Chow"              colored --rhie-chow
run smooth  "residual + RC, boundary, transient" colored --rhie-chow --boundary --transient 0.01
run smooth  "Jex bare"                          colored --jac-action
run smooth  "Jex + Rhie-Chow"                   colored --jac-action --rhie-chow
run smooth  "Jlag (Venkatakrishnan lagged)"     colored --jac-action --rhie-chow --conv-ho 2 --lagged-ho

# The f32 build refuses the double oracles rather than running them against double bounds.
if "$B32" --n 6 --repeat 1 --warmup 0 --verify-ho --layout atomic --conv-ho 2 > /dev/null 2>&1; then
    printf '%-42s FAIL (accepted a double-precision oracle)\n' "f32 build refuses --verify-ho"
    FAIL=$((FAIL + 1))
else
    printf '%-42s OK\n' "f32 build refuses --verify-ho"
fi

if [ "$FAIL" -ne 0 ]; then echo "cvfem_bench_f32: $FAIL FAILED"; exit 1; fi
echo "cvfem_bench_f32: PASSED"
