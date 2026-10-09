#!/usr/bin/env bash
# THE GEOMETRY GATE, ON A MESH THAT IS NOT A CUBE.
#
# Every other oracle in this suite runs on a uniform cube, and a cube cannot see this class of
# bug. On a perfect cube the affine edge vector -- column q of the element Jacobian -- and the
# difference of the two nodes' coordinates are the same number to the last bit, so a kernel using
# one and its reference using the other agree perfectly and the check passes while the two are
# discretising different geometries.
#
# They part company as soon as the mesh is not exactly affine, and the Rhie-Chow correction
# amplifies the parting: it is a near-cancellation, (p_j - p_i) minus the gradient-implied
# difference, so a last-bit difference in the edge vector becomes a visible relative error in the
# term. That is how this escaped -- the vectorised kernels were moved to the Jacobian column and
# the scalar reference was left on coordinates, every cube-based check passed, and the
# disagreement surfaced only at n=160 in a campaign, at 2.0e-4, because that was the first size
# whose coordinates did not land exactly.
#
# So this gate warps the mesh. It is cheap, it runs at small n, and it would have caught that
# the day it was introduced.
set -uo pipefail
BENCH="${1:?usage: $0 <path to cvfem_hex8_ns_upwind_bench>}"
N="${N:-20}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Round-off, not a discretisation tolerance: these compare two evaluations of the SAME operator,
# so they must agree to machine epsilon however distorted the mesh is.
TOL="1e-12"
fail=0

check() { # label value
    awk -v l="$1" -v v="$2" -v t="$TOL" 'BEGIN{
        ok = (v+0 < t+0) ? "OK" : "FAIL";
        printf "  %-46s %12s   %s\n", l, v, ok;
        exit (ok == "OK") ? 0 : 1 }'
}

for warp in 0 0.05 0.2; do
    echo "warp=$warp"

    out=$("$BENCH" --n "$N" --repeat 1 --warmup 0 --layout packed \
                   --jac-action --rhie-chow --warp "$warp" 2>&1)
    rel=$(printf '%s\n' "$out" | sed -n 's/.*jac_action_rc_vs_atomic_rel: \([0-9.e+-]*\).*/\1/p' | head -1)
    if [ -z "$rel" ]; then
        echo "  Jacobian cross-layout                           (no value)   FAIL"
        printf '%s\n' "$out" | tail -3
        fail=1
    else
        check "Jacobian vs the scalar reference" "$rel" || fail=1
    fi

    out=$("$BENCH" --n 24 --layout packed --rhie-chow --warp "$warp" --verify 2>&1)
    abs=$(printf '%s\n' "$out" | sed -n 's/.*verify_rc_packed_residual_vs_atomic_abs: \([0-9.e+-]*\).*/\1/p' | head -1)
    if [ -z "$abs" ]; then
        echo "  residual packed vs atomic                       (no value)   FAIL"
        fail=1
    else
        check "residual packed vs atomic" "$abs" || fail=1
    fi

    # The element-coloured layout renumbers the elements before the adjugate, the determinant and
    # the Rhie-Chow surface tables are built. If that ordering is ever broken -- the tables built
    # first and permuted after, or not permuted at all -- the kernel and its inputs describe
    # different elements, and on a cube that is invisible because the adjugate is the same for
    # every element. Here it is not.
    out=$("$BENCH" --n 24 --layout ecolor --rhie-chow --warp "$warp" --verify 2>&1)
    abs=$(printf '%s\n' "$out" | sed -n 's/.*verify_ecolor_residual_vs_atomic_abs: \([0-9.e+-]*\).*/\1/p' | head -1)
    if [ -z "$abs" ]; then
        echo "  residual ecolor vs atomic                       (no value)   FAIL"
        fail=1
    else
        check "residual ecolor vs atomic" "$abs" || fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "cvfem_warped_geometry: a kernel and its reference disagree on a non-affine mesh."
    echo "That is a geometry inconsistency, not a tolerance: every producer of precomputed"
    echo "geometry and every consumer of it must use the SAME edge vector. See"
    echo "cvfem_hex8_affine_edge_cols."
    exit 1
fi
echo "all warped-geometry checks passed"
