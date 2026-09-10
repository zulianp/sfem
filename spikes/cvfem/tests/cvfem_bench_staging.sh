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

# A configuration the driver must still refuse, because no kernel behind it carries the term
# the flags asked for. A refusal is a feature: the alternative is a row that names a term it
# did not compute.
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
differs() {
    desc="$1"; shift
    off=$("$BENCH" --n 8 --repeat 1 --warmup 0 --layout atomic "$@" 2>&1 | sed -n 's/^ *checksum: //p')
    on=$("$BENCH" --n 8 --repeat 1 --warmup 0 --layout atomic "$@" --rhie-chow 2>&1 | sed -n 's/^ *checksum: //p')
    if [ -z "$off" ] || [ -z "$on" ]; then
        printf '%-62s FAIL (a run produced no checksum: off=%s on=%s)\n' "$desc" "${off:-none}" "${on:-none}"
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
ok "assemble + boundary, sumfact"              --assemble --boundary --kernel sumfact
ok "assemble + boundary, sympy"                --assemble --boundary --kernel sympy
ok "assemble + boundary, sympy_block"          --assemble --boundary --kernel sympy_block
ok "assemble + boundary, split"                --assemble --boundary --kernel split
ok "assemble + boundary, fd"                   --assemble --boundary --kernel fd
ok "assemble + boundary, isoparam current"     --assemble --boundary --geom isoparam --kernel current
ok "assemble + boundary, isoparam sympy"       --assemble --boundary --geom isoparam --kernel sympy
ok "assemble + boundary, isoparam split"       --assemble --boundary --geom isoparam --kernel split
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
ok "assemble, isoparam split + rc"             --assemble --geom isoparam --kernel split --rhie-chow
differs "residual, isoparam current"           --geom isoparam --kernel current
differs "jac-action, isoparam"                 --jac-action --geom isoparam
differs "assemble, isoparam current"           --assemble --geom isoparam --kernel current
differs "assemble, isoparam split"             --assemble --geom isoparam --kernel split
differs "assemble, affine split"               --assemble --kernel split
differs "block diagonal, affine"               --assemble-diag
differs "block diagonal, isoparam"             --assemble-diag --geom isoparam

# The assembled matrix with Rhie-Chow on, across layouts. There is no external reference
# that carries the term, so the check is that the four layouts produce the same matrix: they
# all run the same scalar element kernel and differ only in how the result is scattered.
# The checksums agree to about 1e-15 relative rather than exactly, because the scatter order
# differs -- so this compares numerically, not as strings.
same_across_layouts() {
    desc="$1"; shift
    ref=""
    for lay in atomic packed colored store; do
        v=$("$BENCH" --n 8 --repeat 1 --warmup 0 --layout "$lay" "$@" 2>&1 | sed -n 's/^ *checksum: //p')
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
ok "split + transient, isoparam"               --assemble --geom isoparam --kernel split --transient 0.01
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

echo "== still refused, and must stay so"
# No generated kernel carries Rhie-Chow: the term was never put into the SymPy expressions.
refused "assemble + rc, sympy"                 --assemble --rhie-chow --kernel sympy
refused "assemble + rc, sympy_block"           --assemble --rhie-chow --kernel sympy_block
refused "assemble + rc, isoparam sympy"        --assemble --rhie-chow --geom isoparam --kernel sympy
# The finite-difference reference differences a residual that takes no pressure gradient.
refused "assemble + rc, fd"                    --assemble --rhie-chow --kernel fd
# The hand-written affine residual takes no Hex8RhieChow.
refused "residual + rc, current"               --rhie-chow --kernel current
# The isoparametric SIMD kernels take none either, which is what confines the
# isoparametric case to the atomic layout.
refused "residual + rc, isoparam packed"       --rhie-chow --geom isoparam --kernel current --layout packed
refused "jac-action + rc, isoparam packed"     --rhie-chow --jac-action --geom isoparam --layout packed
# Assembly on the pack-based layouts has no Rhie-Chow staging at all yet.
# Isoparametric residual and action on a pack-based layout run the SIMD kernels, which
# carry no term. Assembly there is scalar and does, which is why only these two are refused.
refused "residual + rc, isoparam store"        --rhie-chow --geom isoparam --kernel current --layout store
refused "jac-action + rc, isoparam colored"    --rhie-chow --jac-action --geom isoparam --layout colored
# The generated action arrangements carry no boundary or Rhie-Chow term.
refused "jac-action + rc, sympy_action"        --jac-action --rhie-chow --kernel sympy_action

if [ "$FAIL" -ne 0 ]; then
    echo "cvfem_bench_staging: $FAIL configuration(s) failed"
    exit 1
fi
echo "all staged configurations agree, all refused configurations still refuse"
