#!/bin/sh
# What the benchmark stages, and what it still refuses.
#
# Stage B of the kernel-coherence work gave the assembled operator its boundary closure for
# every kernel and layout, gave the block diagonal both Rhie-Chow and the closure, and gave
# the atomic isoparametric paths Rhie-Chow. Each of those turned a refusal into a number,
# and a number is only worth having if something checks it -- so every combination that was
# opened is run here against the driver's own oracle, and every combination still refused is
# run to confirm it still exits non-zero.
#
# The oracles are the driver's, not this script's:
#
#   verify_jac_spmv_vs_fd_rel          the assembled matrix against a central difference of
#                                      the residual, over the whole mesh. With --boundary on
#                                      this is also the only mesh-level check that
#                                      boundary_scs_add_jacobian is the derivative of
#                                      boundary_scs_add_residual.
#   verify_diag_vs_full_assembly_rel   the block diagonal against the diagonal blocks pulled
#                                      out of the full assembly, with whatever terms the run
#                                      asked for on both sides.
#   verify_split_isoparam_vs_full_rel  linear + nonlinear against the full assembly, which is
#                                      what says Rhie-Chow went into the nonlinear half only.
#
# n=8 throughout: this is a correctness gate, and a bigger mesh would only make it slower.
# It says nothing about speed and must not be read as if it did.

set -u
BENCH="${1:?usage: cvfem_bench_staging.sh <path to cvfem_hex8_ns_upwind_bench>}"
FAIL=0

# A configuration that must run and pass every check the driver makes.
ok() {
    desc="$1"; shift
    # --layout atomic comes first so a case that wants another layout can say so in "$@"
    # and win: the parser assigns on every match, so the last one holds.
    if out=$("$BENCH" --n 8 --repeat 1 --warmup 0 --verify-jac --layout atomic "$@" 2>&1); then
        printf '%-62s OK   %s\n' "$desc" \
            "$(printf '%s\n' "$out" | grep -oE 'verify_(jac_spmv_vs_fd_rel|diag_vs_full_assembly_rel|split_isoparam_vs_full_rel|rc_colored_residual_vs_atomic_abs|jac_mf_colored_action_vs_packed_abs): [0-9.e+-]*' | tr '\n' ' ')"
    else
        printf '%-62s FAIL\n' "$desc"
        printf '%s\n' "$out" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
}

# The higher-order deferred correction. Its two oracles live behind --verify rather than
# --verify-jac, and until now NOTHING in ctest ran them: the checks existed only for whoever
# remembered to pass the flag, so a change to the reconstruction or to how a layout gathers its
# inputs could break equivalence and the suite would stay green. Both oracles abort the driver on
# failure, so a non-zero exit here is the gate.
#
#   verify_packed_ho_residual_vs_atomic_abs      the two LAYOUTS agree
#   verify_packed_ho_simd_vs_packed_ho_scalar_abs   the two hand-written KERNELS agree
#   verify_packed_ho_sympy_vs_packed_ho_scalar_abs  the GENERATED kernel agrees (arm 0 only)
# The first-order checksum, to prove each arm below actually applied its correction. Both
# oracles read exactly 0.0 at n=8 -- one pack, so the layouts see the same element order and
# the same arithmetic -- and an agreement of zero would also hold if the correction were
# silently skipped in both. So agreement is checked AND the answer is checked to have moved.
FO_CK=$("$BENCH" --n 8 --repeat 1 --warmup 0 --layout packed 2>&1 \
        | sed -n 's|^  checksum: \(.*\)$|\1|p' | head -1)

ok_ho() {
    desc="$1"; shift
    if ! out=$("$BENCH" --n 8 --repeat 1 --warmup 0 --verify-ho --layout atomic "$@" 2>&1); then
        printf '%-62s FAIL\n' "$desc"
        printf '%s\n' "$out" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        return
    fi
    ck=$("$BENCH" --n 8 --repeat 1 --warmup 0 --layout packed "$@" 2>&1 \
         | sed -n 's|^  checksum: \(.*\)$|\1|p' | head -1)
    if [ -z "$ck" ] || [ "$ck" = "$FO_CK" ]; then
        printf '%-62s FAIL (checksum equals first order: correction not applied)\n' "$desc"
        FAIL=$((FAIL + 1))
        return
    fi
    printf '%-62s OK   %s\n' "$desc" \
        "$(printf '%s\n' "$out" | grep -oE 'verify_packed_ho_(residual_vs_atomic|simd_vs_packed_ho_scalar|sympy_vs_packed_ho_scalar)_abs: [0-9.e+-]*|verify_packed_ho_(jac_)?f32_storage_vs_f64_rel: [0-9.e+-]*' | tr '\n' ' ')"
}

# The Rhie-Chow higher-order oracle. Its own helper because the driver reaches it by a different
# path and prints a different line, and because the checksum-moved assertion above does not apply:
# both sides here carry the correction, so what is being tested is that the two KERNELS agree.
ok_ho_rc() {
    desc="$1"; shift
    if out=$("$BENCH" --n 8 --repeat 1 --warmup 0 --verify-ho --layout packed "$@" 2>&1); then
        printf '%-62s OK   %s\n' "$desc" \
            "$(printf '%s\n' "$out" | grep -oE 'verify_packed_ho_rc_sympy_vs_scalar_abs: [0-9.e+-]*')"
    else
        printf '%-62s FAIL\n' "$desc"
        printf '%s\n' "$out" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
}

# A configuration the driver must still refuse, because no kernel behind it carries the term
# the flags asked for. A refusal is a feature: the alternative is a row that names a term it
# did not compute.
# A REFUSAL THAT THE SUBPAR BUILD LIFTS. Everything quarantined in subpar/ is refused by name
# in the default build and ACCEPTED under -DCVFEM_ENABLE_SUBPAR=ON, which is the whole point of
# the flag -- so an unconditional `refused` assertion is wrong in exactly that build. This was
# invisible for as long as the subpar build did not compile at all; once it did, the PA case
# below failed for being accepted, which is the correct behaviour there.
#
# CVFEM_HAS_SUBPAR is set by CMake from the option, so the script is told which build it is in
# rather than guessing from the binary.
refused_unless_subpar() {
    if [ "${CVFEM_HAS_SUBPAR:-0}" = "1" ]; then
        desc="$1"; shift
        if "$BENCH" --n 8 --repeat 1 --warmup 0 --layout atomic "$@" >/dev/null 2>&1; then
            printf '%-62s OK   accepted (subpar build)\n' "$desc"
        else
            printf '%-62s FAIL (refused in a build that enables it)\n' "$desc"
            FAIL=$((FAIL + 1))
        fi
        return
    fi
    refused "$@"
}
refused() {
    desc="$1"; shift
    if "$BENCH" --n 8 --repeat 1 --warmup 0 --layout atomic "$@" >/dev/null 2>&1; then
        printf '%-62s FAIL (accepted; it carries no such term)\n' "$desc"
        FAIL=$((FAIL + 1))
    else
        printf '%-62s OK   refused\n' "$desc"
    fi
}

# The term reaches the kernel: the same configuration with and without --rhie-chow must not
# produce the same checksum. Both runs must also succeed.
# Does turning a term on change the answer at all?
#
# The bitwise fingerprint, at one thread, rather than the checksum. Two reasons, and the
# first one bit:
#
#   * the checksum is a SIGNED SUM, and on these residuals it cancels to ~2e-14 out of terms
#     of order one. A term whose contribution is small relative to the state can then move
#     every value in the vector and leave that sum looking untouched -- which is exactly what
#     happened once the Rhie-Chow time scale stopped being a factor of Peclet too large. A
#     fold over every value's bit pattern cannot be cancelled against itself.
#   * the atomic layout -- the only one the isoparametric kernels support with this term -- has
#     a thread-order-dependent scatter, so at more than one thread BOTH numbers move on their
#     own and the comparison is measuring noise in either direction. One thread makes the
#     scatter deterministic, which is what lets a difference mean something.
differs() {
    desc="$1"; shift
    off=$(OMP_NUM_THREADS=1 "$BENCH" --n 8 --repeat 1 --warmup 0 --layout atomic "$@" 2>&1 | sed -n 's/^ *fingerprint: //p')
    on=$(OMP_NUM_THREADS=1 "$BENCH" --n 8 --repeat 1 --warmup 0 --layout atomic "$@" --rhie-chow 2>&1 | sed -n 's/^ *fingerprint: //p')
    if [ -z "$off" ] || [ -z "$on" ]; then
        printf '%-62s FAIL (a run produced no fingerprint: off=%s on=%s)\n' "$desc" "${off:-none}" "${on:-none}"
        FAIL=$((FAIL + 1))
    elif [ "$off" = "$on" ]; then
        printf '%-62s FAIL (--rhie-chow changed nothing: %s)\n' "$desc" "$on"
        FAIL=$((FAIL + 1))
    else
        printf '%-62s OK   rc off %s -> on %s\n' "$desc" "$off" "$on"
    fi
}

echo "== the boundary closure now reaches every assembly kernel and geometry"
ok "assemble, no terms"                        --assemble
ok "assemble + boundary, sumfact"              --assemble --boundary
# Two rows, not eight. Seven of them named a micro-kernel variant; DESIGN.md's correction
# leaves one assembly kernel per geometry, so what is left to check is that the boundary
# closure reaches both of them.
ok "assemble + boundary, isoparam"             --assemble --boundary --geom isoparam
ok "bsr-apply + boundary"                      --bsr-apply --boundary

echo "== the block diagonal carries both terms, on both geometries"
ok "diag, no terms"                            --assemble-diag
ok "diag + rhie-chow"                          --assemble-diag --rhie-chow
ok "diag + boundary"                           --assemble-diag --boundary
ok "diag + rhie-chow + boundary"               --assemble-diag --rhie-chow --boundary
ok "diag, isoparam"                            --assemble-diag --geom isoparam
ok "diag + both, isoparam"                     --assemble-diag --geom isoparam --rhie-chow --boundary

echo "== Rhie-Chow on the atomic isoparametric paths"
# These have no oracle to compare against -- the reference kernels carry no Rhie-Chow, which
# is why --verify-jac refuses the combination. What can be checked, and is the thing that
# would actually have been wrong, is that the term reaches the kernel at all: a run with it
# on must not produce the same answer as a run with it off. An argument that is accepted,
# forwarded and then defaulted away is exactly the silent failure this whole stage is about.
# `split` is gone (2.7x slower, Grace job 4982167) and `current` is no longer a name but the
# kernel the isoparametric atomic arms select WHEN RHIE-CHOW IS ON -- which is exactly the
# dispatch these rows now exercise: the term has to reach it, or the run with it on would
# answer the same as the run with it off.
differs "residual, isoparam atomic"            --geom isoparam
differs "jac-action, isoparam"                 --jac-action --geom isoparam
differs "assemble, isoparam atomic"            --assemble --geom isoparam
differs "block diagonal, affine"               --assemble-diag
differs "block diagonal, isoparam"             --assemble-diag --geom isoparam

# The assembled matrix with Rhie-Chow on, across layouts. There is no external reference
# that carries the term, so the check is that the four layouts produce the same matrix: they
# all run the same scalar element kernel and differ only in how the result is scattered.
# The checksums agree to about 1e-15 relative rather than exactly, because the scatter order
# differs -- so this compares numerically, not as strings.
# At one thread, for the reason the differs() comment gives: `atomic` is the reference here
# and its scatter is thread-order-dependent, so above one thread the reference itself moves
# between the four runs being compared against it. The checksum also cancels to ~5e-12 out of
# terms of order one on the Rhie-Chow configurations, which turns that movement into a
# relative difference of 1e-4 on a comparison made at 1e-12. Pinning the thread count makes
# all four layouts agree bit for bit, which is the honest form of this check.
same_across_layouts() {
    desc="$1"; shift
    ref=""
    for lay in atomic packed colored store; do
        # THE PACK SIZE IS PINNED for the same reason the thread count is. This check wants one
        # summation order across the four layouts, and at --n 8 it used to get it for free: the
        # old fixed default of 2048 elements put all 512 elements of that mesh in a single pack,
        # so the packed sweep summed in the flat order. The default now derives from the core
        # count (cvfem_default_pack_size), which at one thread gives 32 packs and therefore a
        # different order and a different last bit -- and since this checksum cancels to ~5e-12
        # out of terms of order one, a last-bit move is a large RELATIVE move. Asking for one
        # pack explicitly keeps the check testing the layouts rather than the packer's default.
        v=$(OMP_NUM_THREADS=1 "$BENCH" --n 8 --repeat 1 --warmup 0 --pack-size 8192 --layout "$lay" "$@" 2>&1 | sed -n 's/^ *checksum: //p')
        if [ -z "$v" ]; then
            printf '%-62s FAIL (--layout %s produced no checksum)\n' "$desc" "$lay"
            FAIL=$((FAIL + 1))
            return
        fi
        if [ -z "$ref" ]; then ref="$v"; continue; fi
        if ! awk -v a="$ref" -v b="$v" 'BEGIN{d=a-b; if(d<0)d=-d; s=a; if(s<0)s=-s; if(s==0)s=1; exit !(d <= 1e-12*s)}'; then
            printf '%-62s FAIL (--layout %s gave %s, atomic gave %s)\n' "$desc" "$lay" "$v" "$ref"
            FAIL=$((FAIL + 1))
            return
        fi
    done
    printf '%-62s OK   four layouts agree on %s\n' "$desc" "$ref"
}

echo "== Rhie-Chow in the assembled matrix, on every layout"
same_across_layouts "assemble + rc"                --assemble --rhie-chow
same_across_layouts "assemble + rc + boundary"     --assemble --rhie-chow --boundary
same_across_layouts "assemble + rc + transient"    --assemble --rhie-chow --transient 0.01
same_across_layouts "assemble, no terms"           --assemble

echo "== Rhie-Chow on the pack-based layouts"
# The oracle is this driver's own implementations against each other: packed, colored and
# atomic are three spellings of one matrix-free operator, so with the term on they must
# agree to round-off. That is the only check that says the colored sweep stages Rhie-Chow
# the way the packed one does -- there is no external reference that carries the term.
ok "residual + rc, packed"                     --rhie-chow --layout packed
ok "residual + rc, colored"                    --rhie-chow --layout colored
ok "residual + rc, store"                      --rhie-chow --layout store
ok "jac-action + rc, packed"                   --rhie-chow --jac-action --layout packed
ok "jac-action + rc, colored"                  --rhie-chow --jac-action --layout colored
# The element-coloured layout is not in same_across_layouts above and must not be: each node
# takes its contributions one per colour, so the summation order differs from the atomic sweep's
# and the checksums differ in the last bits by construction. Its gate is the per-node comparison
# this runs, plus the geometry-consistency check in tests/cvfem_warped_geometry.sh.
ecolor_ok() { # desc, extra args -- --verify rather than --verify-jac, see the driver's refusal
    desc="$1"; shift
    if out=$("$BENCH" --n 8 --repeat 1 --warmup 0 --verify --layout ecolor "$@" 2>&1); then
        printf '%-62s OK   %s\n' "$desc" \
            "$(printf '%s\n' "$out" | grep -oE 'verify_ecolor_[a-z_]*: [0-9.e+-]*' | tr '\n' ' ')"
    else
        printf '%-62s FAIL\n' "$desc"
        printf '%s\n' "$out" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
}
ecolor_ok "residual + rc, ecolor"              --rhie-chow
ecolor_ok "jac-action + rc, ecolor"            --rhie-chow --jac-action

echo "== the transient term, on every operation"
# tests/cvfem_bench_transient_test pins the term itself against closed forms. What is
# checked here is that it REACHES each operation and each layout, which is a property of the
# driver rather than of the pass.
ok "residual + transient"                      --transient 0.01
ok "residual + transient, BDF2"                --transient 0.01 --bdf 2
ok "jac-action + transient"                    --jac-action --transient 0.01
ok "assemble + transient"                      --assemble --transient 0.01
ok "assemble + transient + boundary"           --assemble --transient 0.01 --boundary
ok "diag + transient"                          --assemble-diag --transient 0.01
ok "diag + transient + rc + boundary"          --assemble-diag --transient 0.01 --rhie-chow --boundary
ok "assemble + transient, isoparam"            --assemble --geom isoparam --transient 0.01
transient_reaches() {
    desc="$1"; shift
    off=$("$BENCH" --n 8 --repeat 1 --warmup 0 --layout "$@" 2>&1 | sed -n 's/^ *checksum: //p')
    on=$("$BENCH" --n 8 --repeat 1 --warmup 0 --layout "$@" --transient 0.01 2>&1 | sed -n 's/^ *checksum: //p')
    if [ -n "$off" ] && [ -n "$on" ] && [ "$off" != "$on" ]; then
        printf '%-62s OK   steady %s -> transient %s\n' "$desc" "$off" "$on"
    else
        printf '%-62s FAIL (off=%s on=%s)\n' "$desc" "${off:-none}" "${on:-none}"
        FAIL=$((FAIL + 1))
    fi
}
transient_reaches "residual, packed"           packed
transient_reaches "residual, colored"          colored
transient_reaches "jac-action, packed"         packed --jac-action
transient_reaches "assemble, store"            store --assemble

echo "== the higher-order deferred correction, every limiter arm"
# All four arms, because the limiter is the part of the reconstruction with data-dependent
# selects in it and an arm can break on its own.
ok_ho "residual + ho, unlimited"                --conv-ho 0
ok_ho "residual + ho, bounded-face clip"        --conv-ho 1
ok_ho "residual + ho, Venkatakrishnan"          --conv-ho 2
ok_ho "residual + ho, Darwish-Moukalled"        --conv-ho 3
# The mixed-precision storage option: the same correction with the nodal gradient held as float.
# Its oracle is the f32-against-f64 line the driver prints beside the layout check.
ok_ho "residual + ho, Venkatakrishnan, f32 gradient storage" --conv-ho 2 --grad-precision single
ok_ho "residual + ho, unlimited, f32 gradient storage"       --conv-ho 0 --grad-precision single

# THE LIMITED JACOBIANS' LAYOUT AGREEMENT AT THE SIZES THAT BROKE IT. The driver checks the packed
# exact action against the atomic reference on every Rhie-Chow Jacobian run and refuses on a
# mismatch, but every gate above runs at n=8 or 10 and every throughput job at 128 -- sizes where
# it held -- while gcc builds disagreed by 1e-4 at n=24, 40, 72, 80, 96 and 160 until the limiter
# derivatives' band was scaled by the increment it tests and widened to 64 rounding units (see
# cvfem_venkata_limiter.hpp). Three of those sizes, all three limiters: a non-zero exit is the
# driver's own refusal.
for lim_n in 24 40 72; do
    for lim in 1 2 3; do
        desc="Jhoex packed vs atomic, limiter $lim, n=$lim_n"
        if out=$("$BENCH" --n $lim_n --repeat 1 --warmup 0 --layout packed --jac-action --rhie-chow \
                     --conv-ho $lim 2>&1); then
            printf '%-62s OK   %s\n' "$desc" "$(printf '%s\n' "$out" | grep -oE 'jac_action_rc_vs_atomic_rel: [0-9.e+-]*')"
        else
            printf '%-62s FAIL\n' "$desc"
            printf '%s\n' "$out" | grep -E 'disagrees|rel' | sed 's/^/    /'
            FAIL=$((FAIL + 1))
        fi
    done
done
# Rhie-Chow, where the generated kernel has its own variant. This arm takes a different route
# through the driver -- the main verify block is gated off when Rhie-Chow is on -- so it needs its
# own row rather than being implied by the four above.
ok_ho_rc "residual + ho + rc, generated vs scalar" --conv-ho 0 --rhie-chow
echo

# THE HIGHER-ORDER JACOBIAN ACTION, PACKED AGAINST ATOMIC, AT A NON-POWER-OF-TWO SIZE.
#
# Every check above runs the residual, and every check in this file runs --n 8. Both choices hid
# a real defect for as long as the higher-order Jacobian action has existed:
#
#   * the limiters' DERIVATIVES had branches guarded against quantities that vanish with the
#     thing being tested -- Venkatakrishnan's inc >= 0 select, where psi is 0 rather than 1
#     when the base is the bound, and Darwish-Moukalled's band, which was relative to the sum it
#     was bounding. The packed and atomic layouts reach them with inc differing in the last bits,
#     because their nodal-gradient reconstructions sum in different orders, so an ulp picked a
#     side and the two answers differed by 1e-04 to 3e-04.
#   * it is invisible at a POWER-OF-TWO cube size. Measured on Grace with the pre-fix binary:
#     n=6 clean, then n=10, 12, 18, 20, 24, 36, 40, 48, 72 through 112 all failing, and 64 and
#     128 clean again. --n 8 is the one size this file used.
#   * and the residual never sees it, because the limiter's VALUE is continuous where its
#     derivative is not.
#
# The driver's own packed-versus-atomic oracle catches it and exits non-zero. It had simply
# never been run anywhere that it fires. n=10 is the cheapest size that does.
#
# This arm passes trivially on the development machine -- the defect did not reproduce there at
# any size -- so it earns its keep in the Alps ctest run rather than locally.
echo "== the higher-order Jacobian action, packed against atomic, off a power of two"
ho_jac_ok() {
    desc="$1"; shift
    if out=$("$BENCH" --n 10 --repeat 1 --warmup 0 --layout packed --jac-action --rhie-chow "$@" 2>&1); then
        printf '%-62s OK   %s\n' "$desc" \
            "$(printf '%s\n' "$out" | grep -oE 'jac_action_rc_vs_atomic_rel: [0-9.e+-]*' | tr '\n' ' ')"
    else
        printf '%-62s FAIL\n' "$desc"
        printf '%s\n' "$out" | grep -E 'disagrees|_rel:' | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
}
ho_jac_ok "jac-action + ho, unlimited"            --conv-ho 0
ho_jac_ok "jac-action + ho, bounded-face clip"    --conv-ho 1
ho_jac_ok "jac-action + ho, Venkatakrishnan"      --conv-ho 2
ho_jac_ok "jac-action + ho, Darwish-Moukalled"    --conv-ho 3
# And with Venkatakrishnan's eps^2 on, which is the other route to a well-conditioned
# derivative and the one the front end takes.
ho_jac_ok "jac-action + ho, Venkatakrishnan K=5"  --conv-ho 2 --venkat-k 5
echo

echo "== the partially assembled Jacobian action"
# Measured and lost -- 17-19% slower than direct evaluation, see subpar/README.md -- so the
# default build refuses it by name, the way it refuses every other retired kernel.
# Its correctness is still covered, by cvfem_pa_tangent_test, which calls the kernels
# directly and so keeps the quarantined path from rotting.
refused_unless_subpar "PA (quarantined, needs -DCVFEM_ENABLE_SUBPAR=ON)"  --jac-action --layout packed --rhie-chow --partial-assembly

echo "== still refused, and must stay so"
# Eleven rows stood here. Seven refused a configuration by the NAME of a micro-kernel -- a
# generated arrangement or the finite-difference reference asked for with --rhie-chow, which
# none of them carries -- and those names no longer exist: the atomic arms now select the
# kernel that does carry the term, so there is nothing left to refuse. What remains is the
# refusal that is about a LAYOUT rather than a name, and it is the one that still bites: the
# isoparametric residual and action on a pack-based layout run the SIMD kernels, which carry
# no term, so the term cannot be honoured there whatever kernel is chosen.
refused "residual + rc, isoparam packed"       --rhie-chow --geom isoparam --layout packed
refused "jac-action + rc, isoparam packed"     --rhie-chow --jac-action --geom isoparam --layout packed
refused "residual + rc, isoparam store"        --rhie-chow --geom isoparam --layout store
refused "jac-action + rc, isoparam colored"    --rhie-chow --jac-action --geom isoparam --layout colored
# Pack colouring has no higher-order correction in either operator; before the refusal the
# residual ran first order under the --conv-ho label. The lagged action needs no correction and
# must still run there.
refused "residual + ho, colored"                --rhie-chow --conv-ho 3 --layout colored
refused "jac-action + ho exact, colored"        --rhie-chow --jac-action --conv-ho 3 --layout colored
ok "jac-action + ho lagged, colored"            --rhie-chow --jac-action --conv-ho 3 --lagged-ho --layout colored

if [ "$FAIL" -ne 0 ]; then
    echo "cvfem_bench_staging: $FAIL configuration(s) failed"
    exit 1
fi
echo "all staged configurations agree, all refused configurations still refuse"
