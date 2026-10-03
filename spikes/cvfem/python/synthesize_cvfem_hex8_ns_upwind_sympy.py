#!/usr/bin/env python3
"""Generate SymPy/CSE CVFEM HEX8 Navier-Stokes kernels."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import sympy as sp

from cvfem_codegen import (
    N_FIELD,
    ScalarPrinter,
    add_output_arguments,
    cse_emit,
    dof,
    emit,
    face_flux_residual,
    jac_block_exprs as _jac_block_exprs,
    sign_locals as _sign_locals,
)


HERE = Path(__file__).resolve().parent
# python/ -> spike root. The emitted headers live with the sources they are
# compiled into, not beside the generator that writes them.
SPIKE_ROOT = HERE.parent
OUT = SPIKE_ROOT / "src" / "kernels" / "microkernels" / "hex8" / "generated" / "cvfem_hex8_ns_upwind_sympy_kernels.hpp"

N_NODE = 8
N_DOF = N_NODE * N_FIELD

SCS = (
    (0, 1, (sp.Rational(1, 4), sp.Rational(0), sp.Rational(0))),
    (3, 2, (sp.Rational(1, 4), sp.Rational(0), sp.Rational(0))),
    (4, 5, (sp.Rational(1, 4), sp.Rational(0), sp.Rational(0))),
    (7, 6, (sp.Rational(1, 4), sp.Rational(0), sp.Rational(0))),
    (0, 3, (sp.Rational(0), sp.Rational(1, 4), sp.Rational(0))),
    (1, 2, (sp.Rational(0), sp.Rational(1, 4), sp.Rational(0))),
    (4, 7, (sp.Rational(0), sp.Rational(1, 4), sp.Rational(0))),
    (5, 6, (sp.Rational(0), sp.Rational(1, 4), sp.Rational(0))),
    (0, 4, (sp.Rational(0), sp.Rational(0), sp.Rational(1, 4))),
    (1, 5, (sp.Rational(0), sp.Rational(0), sp.Rational(1, 4))),
    (2, 6, (sp.Rational(0), sp.Rational(0), sp.Rational(1, 4))),
    (3, 7, (sp.Rational(0), sp.Rational(0), sp.Rational(1, 4))),
)

DN_REF = (
    (-sp.Rational(1, 4), -sp.Rational(1, 4), -sp.Rational(1, 4)),
    ( sp.Rational(1, 4), -sp.Rational(1, 4), -sp.Rational(1, 4)),
    ( sp.Rational(1, 4),  sp.Rational(1, 4), -sp.Rational(1, 4)),
    (-sp.Rational(1, 4),  sp.Rational(1, 4), -sp.Rational(1, 4)),
    (-sp.Rational(1, 4), -sp.Rational(1, 4),  sp.Rational(1, 4)),
    ( sp.Rational(1, 4), -sp.Rational(1, 4),  sp.Rational(1, 4)),
    ( sp.Rational(1, 4),  sp.Rational(1, 4),  sp.Rational(1, 4)),
    (-sp.Rational(1, 4),  sp.Rational(1, 4),  sp.Rational(1, 4)),
)


# Sub-control-surface reference points, matching CVFEM_HEX8_SCS_XI in the C++ header.
# The reference element is the unit cube, so these are exact rationals.
_H = sp.Rational(1, 2)
_Q = sp.Rational(1, 4)
_T = sp.Rational(3, 4)
SCS_XI = (
    (_H, _Q, _Q), (_H, _T, _Q), (_H, _Q, _T), (_H, _T, _T),
    (_Q, _H, _Q), (_T, _H, _Q), (_Q, _H, _T), (_T, _H, _T),
    (_Q, _Q, _H), (_T, _Q, _H), (_T, _T, _H), (_Q, _T, _H),
)


# Reference coordinates of the eight nodes on the unit cube, in the element's own node order.
# Matches CVFEM_HEX8_REF_XI in the C++ header, and stated here for the same reason that table
# states it rather than deriving it from the DN_REF sign pattern: a reconstruction that reads
# these transposed is wrong in a way no residual norm reveals, because the error is a smooth
# field rather than a blow-up.
REF_XI = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
)


def scs_shape(s: int, a: int) -> sp.Expr:
    """Node ``a``'s trilinear shape function at sub-control surface ``s``'s centroid.

    Exact rationals, products of 1/4, 1/2 and 3/4, which is why the generated centroid is
    bit-identical to the hand-written kernel's -- the same values reach the same FMA chain.
    Derived from the two tables rather than tabulated a third time.
    """
    out = sp.Integer(1)
    for d in range(3):
        out *= SCS_XI[s][d] if REF_XI[a][d] == 1 else (1 - SCS_XI[s][d])
    return out


def dn_ref_at(xi, eta, zeta):
    """Trilinear shape-function derivatives on the unit cube.

    Mirrors cvfem_hex8_dn_ref exactly. The affine generator uses the element centre
    for every face, which is what makes one velocity gradient serve all twelve; the
    isoparametric one evaluates this at each sub-control surface instead, and that
    is the whole of the difference between the two.
    """
    x0 = 1 - xi
    y0 = 1 - eta
    z0 = 1 - zeta
    return (
        (-y0 * z0,   -x0 * z0,   -x0 * y0),
        ( y0 * z0,   -xi * z0,   -xi * y0),
        ( eta * z0,   xi * z0,   -xi * eta),
        (-eta * z0,   x0 * z0,   -x0 * eta),
        (-y0 * zeta, -x0 * zeta,  x0 * y0),
        ( y0 * zeta, -xi * zeta,  xi * y0),
        ( eta * zeta, xi * zeta,  xi * eta),
        (-eta * zeta, x0 * zeta,  x0 * eta),
    )


def build_symbols() -> dict[str, object]:
    return {
        "rho": sp.Symbol("rho"),
        "mu": sp.Symbol("mu"),
        "det": sp.Symbol("det"),
        "cof": sp.symbols("cof0:9"),
        # Isoparametric: one adjugate and determinant per sub-control surface. The
        # caller evaluates them with cvfem_hex8_geom_at; the generated body treats
        # them as opaque symbols exactly as the affine body treats cof/det.
        "cof_s": tuple(sp.symbols(f"c{s}_0:9") for s in range(12)),
        "det_s": tuple(sp.Symbol(f"d{s}") for s in range(12)),
        "ux": sp.symbols("ux0:8"),
        "uy": sp.symbols("uy0:8"),
        "uz": sp.symbols("uz0:8"),
        "p": sp.symbols("p0:8"),
        "sgn": sp.symbols("sgn0:12"),
        # The Krylov direction, for the Jacobian ACTION. The action is the Jacobian
        # contracted with it, so these appear linearly and CSE sees them as ordinary
        # inputs -- the same way ux/uy/uz do.
        "vx": sp.symbols("vx0:8"),
        "vy": sp.symbols("vy0:8"),
        "vz": sp.symbols("vz0:8"),
        "q": sp.symbols("q0:8"),
        # The deferred correction's inputs. The element's node coordinates -- the
        # reconstruction works in physical space -- and the eight nodal velocity gradients,
        # nine components each, g[a][r*3 + c] = du_r/dx_c, matching the layout
        # CVFEMNavierStokes::nodal_velocity_gradient produces and the hand-written kernel reads.
        "x": sp.symbols("x0:8"),
        "y": sp.symbols("y0:8"),
        "z": sp.symbols("z0:8"),
        "g": tuple(sp.symbols(f"g{a}_0:9") for a in range(N_NODE)),
        # The DIRECTION's nodal velocity gradient, same layout as "g". The exact Jacobian
        # action needs it because the reconstruction reads a gradient that is an input to the
        # apply rather than a function of the element's own unknowns: differentiating
        # grad(u_D) . (x_scs - x_D) in direction v gives grad(v_D) . (x_scs - x_D), and
        # grad(v_D) can only come from a reconstruction pass over v. That pass is what makes
        # the exact higher-order action cost more than the lagged one; see the driver's
        # --conv-ho-exact.
        "gv": tuple(sp.symbols(f"gv{a}_0:9") for a in range(N_NODE)),
        # Rhie-Chow. The per-sub-control-surface coefficient arrives as a SYMBOL rather than
        # being built here, and that is deliberate: cvfem_hex8_rhie_chow_mdot_coeff carries two
        # square roots, a division and a data-dependent degeneracy guard that returns zero, so
        # expressing it inline would put a Piecewise in the middle of every surface's flux. The
        # tree already precomputes it per element (cvfem_hex8_build_rc_coeff) and stages it into
        # the pack (cvfem_hex8_gather_rc_coeff), which is the same reason the hand-written
        # Jacobian-action kernels read rc->coeff[s][lane] instead of recomputing it -- the guard
        # inside a vector loop was measured at 1.83x. Reusing that staging keeps this body
        # branch-free and adds no second path.
        "rcoef": sp.symbols("rcoef0:12"),
        "pgx": sp.symbols("pgx0:8"),
        "pgy": sp.symbols("pgy0:8"),
        "pgz": sp.symbols("pgz0:8"),
    }


def area(sym: dict[str, object], ar: tuple[sp.Rational, sp.Rational, sp.Rational],
         face: int | None = None) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    cof = sym["cof"] if face is None else sym["cof_s"][face]
    ar0, ar1, ar2 = ar
    return (
        cof[0] * ar0 + cof[3] * ar1 + cof[6] * ar2,
        cof[1] * ar0 + cof[4] * ar1 + cof[7] * ar2,
        cof[2] * ar0 + cof[5] * ar1 + cof[8] * ar2,
    )


def velocity_gradient(sym: dict[str, object], face: int | None = None) -> tuple[sp.Expr, ...]:
    """Affine (face=None) evaluates once at the element centre and every face shares it.

    Isoparametric evaluates per face, with that face's geometry and that face's shape
    derivatives, so nothing is shared between faces -- which is exactly why the
    isoparametric body is bigger and why CSE has less to work with.
    """
    cof = sym["cof"] if face is None else sym["cof_s"][face]
    det = sym["det"] if face is None else sym["det_s"][face]
    dn = DN_REF if face is None else dn_ref_at(*SCS_XI[face])
    inv_det = 1 / det
    out: list[sp.Expr] = []
    for comp in (sym["ux"], sym["uy"], sym["uz"]):
        dr = sp.Integer(0)
        ds = sp.Integer(0)
        dt = sp.Integer(0)
        for a in range(N_NODE):
            dr += comp[a] * dn[a][0]
            ds += comp[a] * dn[a][1]
            dt += comp[a] * dn[a][2]
        out.extend(
            (
                (cof[0] * dr + cof[3] * ds + cof[6] * dt) * inv_det,
                (cof[1] * dr + cof[4] * ds + cof[7] * dt) * inv_det,
                (cof[2] * dr + cof[5] * ds + cof[8] * dt) * inv_det,
            )
        )
    return tuple(out)


def face_residual_expr(sym: dict[str, object], s: int, isoparam: bool = False) -> tuple[list[sp.Expr], sp.Expr]:
    """The flux across sub-control surface ``s``.

    Only the geometry is HEX8-specific: the area vector and the velocity gradient come
    either from the element-affine adjugate or, under ``isoparam``, from the adjugate
    evaluated at that surface. The flux algebra itself is shared with TET4.
    """
    i, j, ar = SCS[s]
    face = s if isoparam else None
    return face_flux_residual(
        n_dof=N_DOF,
        node_i=i,
        node_j=j,
        area=area(sym, ar, face),
        grad=velocity_gradient(sym, face),
        rho=sym["rho"],
        mu=sym["mu"],
        u=(sym["ux"], sym["uy"], sym["uz"]),
        p=sym["p"],
        sign=sym["sgn"][s],
    )


def residual_exprs(sym: dict[str, object], isoparam: bool = False) -> tuple[list[sp.Expr], list[sp.Expr]]:
    r = [sp.Integer(0)] * N_DOF
    mdots: list[sp.Expr] = []
    for s in range(len(SCS)):
        fr, mdot = face_residual_expr(sym, s, isoparam)
        mdots.append(mdot)
        for i in range(N_DOF):
            r[i] += fr[i]
    return r, mdots


def limit_inc(limiter: int, base: sp.Expr, other: sp.Expr, inc: sp.Expr) -> sp.Expr:
    """One node's reconstruction increment, limited, matching src/venkata/cvfem_venkata_limiter.hpp.

    ``base`` is the donor node's value and ``other`` the surface's far node, so the bound is the
    interval the surface's own two nodes span -- the same one-dimensional stencil the scalar kernel
    uses. Every arm is a select rather than a branch, as it is there.

    Venkatakrishnan's eps^2 term is ZERO here, not omitted by oversight: it is `venkat_c * h^3`
    with h the edge length, so carrying it would put a square root at every surface, and every
    caller in the tree passes venkat_c = 0. The sweep refuses a non-zero venkat_c with the
    generated kernel rather than silently dropping it.
    """
    if limiter == 0:
        return inc
    lo, hi = sp.Min(base, other), sp.Max(base, other)
    if limiter == 1:
        # Bounded face: clip the reconstructed value into the nodal interval, return the increment
        # that survives. Pure Min/Max, no division, nothing for CSE to lose its way in.
        return sp.Max(lo, sp.Min(hi, base + inc)) - base
    if limiter == 2:
        dp  = sp.Piecewise((hi - base, sp.Ge(inc, 0)), (lo - base, True))
        num = dp * dp + 2 * inc * dp
        den = dp * dp + 2 * inc * inc + inc * dp
        # The guard is on `inc`, NOT on `den`, and that is what makes this kernel vectorise.
        #
        # `den` contains `dp`, which is a Piecewise. Writing Ne(den, 0) therefore builds a
        # relational around a Piecewise, and SymPy folds the comparison inside it, so the printer
        # emits a ternary whose CONDITION is another ternary:
        #     ((c) ? (...) : (x70 - x76*x80 + x77 != 0)) ? A : B
        # GCC if-converts the inner conditional into a PHI node and then reports
        #     not vectorized: relevant stmt not supported: iftmp = _c ? iftmp : iftmp
        # and abandons the loop. Measured on the emitted object, the Venkatakrishnan kernel held
        # 6000 instructions and not one NEON register, against 25-30% vector for the other three
        # limiters, which have no Piecewise in any condition -- limiter 3's guard divides by
        # 2*(|a|+|b|), an Abs and not a Piecewise, which is exactly why it was unaffected.
        #
        # Guarding on `inc` is the same function, not an approximation. Writing den as
        # (dp + inc/2)^2 + (7/4)*inc^2 shows it is a sum of squares, so den == 0 requires both
        # dp == 0 and inc == 0; and wherever inc == 0 the guarded branch evaluates to
        # (num/den)*inc == 0 as well, so the two arms already agree there. Hence den == 0 is
        # reachable only inside inc == 0, where this returns the same zero the old guard returned
        # as `inc`. For inc != 0, den >= (7/4)*inc^2 > 0 and the division is safe.
        return sp.Piecewise(((num / den) * inc, sp.Ne(inc, 0)), (sp.Integer(0), True))
    if limiter == 3:
        a, b = 2 * inc, other - base
        aa, ab = sp.Abs(a), sp.Abs(b)
        den = 2 * (aa + ab)
        return sp.Piecewise(((a * ab + aa * b) / den, sp.Ne(den, 0)), (sp.Integer(0), True))
    raise ValueError(f"unknown limiter {limiter}")


def defcor_face_expr(sym: dict[str, object], s: int, mdot: sp.Expr,
                     limiter: int = 0) -> tuple[int, int, list[sp.Expr]]:
    """The deferred-correction momentum increment across sub-control surface ``s``.

    The face value is reconstructed from each donor node and its nodal velocity gradient, and
    only the DIFFERENCE from the first-order upwind value enters, which is what leaves the
    Jacobian first-order. Unlimited: the limiter arms carry data-dependent selects that CSE does
    not see through, and arm 0 is the documented default, so it is the arm generated first.

    ``mdot`` is the surface's mass flux as the flux itself built it, so the correction is
    weighted by the flux actually transported -- including its Rhie-Chow part where present --
    rather than by a second opinion about it. The upwind split below is structurally identical to
    the flux's, so ``sp.cse`` unifies the two rather than emitting both: one expression of the
    split reaches the generated text even though it is written twice here.
    """
    i, j, _ar = SCS[s]
    mdot_abs = sym["sgn"][s] * mdot
    mpos = sp.Rational(1, 2) * (mdot + mdot_abs)
    mneg = sp.Rational(1, 2) * (mdot - mdot_abs)

    x, y, z, g = sym["x"], sym["y"], sym["z"], sym["g"]
    # The surface centroid in physical space. Every face shares these eight coordinates, and
    # generating the whole element as one expression set is what lets CSE hoist the shared part
    # of the twelve centroids -- which twelve separately compiled per-face functions cannot do.
    cen = [sum(scs_shape(s, a) * comp[a] for a in range(N_NODE)) for comp in (x, y, z)]
    di = [cen[0] - x[i], cen[1] - y[i], cen[2] - z[i]]
    dj = [cen[0] - x[j], cen[1] - y[j], cen[2] - z[j]]

    u = (sym["ux"], sym["uy"], sym["uz"])
    out: list[sp.Expr] = []
    for r in range(3):  # velocity component
        inc_i = sum(g[i][r * 3 + k] * di[k] for k in range(3))
        inc_j = sum(g[j][r * 3 + k] * dj[k] for k in range(3))
        # Limited together, against the interval this surface's own two nodes span.
        li = limit_inc(limiter, u[r][i], u[r][j], inc_i)
        lj = limit_inc(limiter, u[r][j], u[r][i], inc_j)
        out.append(mpos * li + mneg * lj)
    return i, j, out


def rc_mdot_expr(sym: dict[str, object], s: int) -> sp.Expr:
    """The Rhie-Chow contribution to surface ``s``'s mass flux.

    The pressure difference across the surface's two nodes, minus the same difference as the
    reconstructed nodal pressure gradients predict it: where the two agree the term vanishes,
    which is what suppresses the checkerboard mode without adding dissipation to a smooth field.
    Scaled by the staged coefficient and subtracted, matching the hand-written kernel's
    ``mdot -= coeff * corr``.
    """
    i, j, _ar = SCS[s]
    x, y, z = sym["x"], sym["y"], sym["z"]
    pgx, pgy, pgz, p = sym["pgx"], sym["pgy"], sym["pgz"], sym["p"]
    dx, dy, dz = x[j] - x[i], y[j] - y[i], z[j] - z[i]
    half = sp.Rational(1, 2)
    corr = (p[j] - p[i]) - (half * (pgx[i] + pgx[j]) * dx +
                            half * (pgy[i] + pgy[j]) * dy +
                            half * (pgz[i] + pgz[j]) * dz)
    return -sym["rcoef"][s] * corr


def residual_defcor_exprs(sym: dict[str, object], rc: bool = False,
                          limiter: int = 0) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """The complete element residual with the deferred correction folded into the momentum rows.

    The correction touches momentum only; the continuity row and the mass flux are untouched,
    which is the whole point of deferring it.
    """
    r = [sp.Integer(0)] * N_DOF
    mdots: list[sp.Expr] = []
    for s in range(len(SCS)):
        i, j, ar = SCS[s]
        fr, mdot = face_flux_residual(
                n_dof=N_DOF, node_i=i, node_j=j, area=area(sym, ar), grad=velocity_gradient(sym),
                rho=sym["rho"], mu=sym["mu"], u=(sym["ux"], sym["uy"], sym["uz"]), p=sym["p"],
                sign=sym["sgn"][s], mdot_rc=rc_mdot_expr(sym, s) if rc else None)
        mdots.append(mdot)
        for k in range(N_DOF):
            r[k] += fr[k]
    for s in range(len(SCS)):
        i, j, inc = defcor_face_expr(sym, s, mdots[s], limiter)
        for c in range(3):
            r[dof(i, c)] += inc[c]
            r[dof(j, c)] -= inc[c]
    return r, mdots


def simd_input_locals_defcor() -> str:
    """Every input this kernel needs, hoisted once per lane from the lane-major packs.

    `pack[node][lane]` over varying lane is contiguous, so each of these is a unit-stride vector
    load, and each value is read ONCE per lane per element rather than once per face. That is the
    property the hand-written kernel had to be coaxed into: it gathered a lane's 96 reconstruction
    inputs into flat locals inside each of the twelve face loops, and then the compiler refused to
    inline the reconstruction and the lane loop stopped vectorising altogether.
    """
    lines = [f"        const scalar_t cof{i} = scalar_t(cof{i}_ptr[lane]);" for i in range(9)]
    lines.append("        const scalar_t det = scalar_t(det_ptr[lane]);")
    for name in ("ux", "uy", "uz", "p"):
        for i in range(N_NODE):
            lines.append(f"        const scalar_t {name}{i} = in.{name}[{i}][lane];")
    for name in ("x", "y", "z"):
        for i in range(N_NODE):
            lines.append(f"        const scalar_t {name}{i} = ho.{name}[{i}][lane];")
    for a in range(N_NODE):
        for c in range(9):
            lines.append(f"        const scalar_t g{a}_{c} = ho.g[{a}][{c}][lane];")
    return "\n".join(lines)


def residual_pack_outputs() -> list[str]:
    """The lane-major residual pack, in this file's dof order: ux, uy, uz, p per node."""
    field = ("rx", "ry", "rz", "rc")
    return [f"out.{field[f]}[{a}][lane]" for a in range(N_NODE) for f in range(N_FIELD)]


def defcor_simd_args() -> str:
    lines = [f"        const scalar_t *const SFEM_RESTRICT cof{i}_ptr," for i in range(9)]
    lines.append("        const scalar_t *const SFEM_RESTRICT det_ptr,")
    return "\n".join(lines)


def simd_rc_locals() -> str:
    """The Rhie-Chow inputs, hoisted per lane from the packs that already hold them.

    The coefficient comes from the pack's staged table, which is what the hand-written
    Jacobian-action kernels read too; the nodal pressure gradients come from the same pack the
    first-order Rhie-Chow path fills. Nothing new is staged for this kernel.
    """
    lines = [f"        const scalar_t rcoef{s} = rc.coeff[{s}][lane];" for s in range(len(SCS))]
    for name in ("pgx", "pgy", "pgz"):
        for i in range(N_NODE):
            lines.append(f"        const scalar_t {name}{i} = rc.{name}[{i}][lane];")
    return "\n".join(lines)


LIMITER_NAME = {
    0: "unlimited reconstruction",
    1: "bounded-face clip",
    2: "Venkatakrishnan's smooth bound, with eps^2 = 0",
    3: "Darwish-Moukalled",
}


def defcor_kernel(sym: dict[str, object], limiter: int, rc: bool) -> str:
    """One lane-blocked higher-order kernel: whole element, one CSE'd block, one loop, no calls."""
    residual, mdots = residual_defcor_exprs(sym, rc=rc, limiter=limiter)
    name = f"cvfem_hex8_ns_upwind_sympy_residual_defcor{'_rc' if rc else ''}_lim{limiter}_simd"
    rc_arg = "        const Hex8RhieChowPack &rc,\n" if rc else ""
    rc_loc = simd_rc_locals() + "\n" if rc else ""
    return f"""
// {LIMITER_NAME[limiter]}{', with Rhie-Chow' if rc else ''}.
static SFEM_INLINE void {name}(
        const scalar_t rho,
        const scalar_t mu,
{defcor_simd_args()}
        const Hex8InputPack &in,
        const Hex8UGradPack &ho,
{rc_arg}        Hex8ResidualPack    &out) {{
#pragma omp simd aligned(cof0_ptr, cof1_ptr, cof2_ptr, cof3_ptr, cof4_ptr, cof5_ptr, cof6_ptr, cof7_ptr, cof8_ptr, det_ptr : 64)
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {{
{simd_input_locals_defcor()}
{rc_loc}{sign_locals(mdots, indent="        ")}
{cse_code(residual, residual_pack_outputs(), indent="        ")}
    }}
}}
"""


def defcor_kernels(sym: dict[str, object]) -> str:
    """All four limiter arms, with and without Rhie-Chow.

    Separate kernels rather than one taking the limiter as an argument: a runtime limiter would put
    a four-way select inside the vector body at every surface and for every velocity component, and
    a sweep-uniform branch in that loop is the shape of guard measured at 1.83x here once already.
    Eight generated functions cost header size and generation time, both of which are recorded in
    the repository's own budget checks, and nothing at run time.
    """
    return "".join(defcor_kernel(sym, lim, rc)
                   for rc in (False, True) for lim in (0, 1, 2, 3))


def defcor_action_exprs(sym: dict[str, object], rc: bool, limiter: int) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """The EXACT Jacobian action of the residual carrying the deferred correction.

    A directional derivative taken term by term rather than a Jacobian built and contracted: the
    correction's Jacobian is dense in a way the first-order one is not -- every momentum row picks
    up the nine gradient components of both of its surface's nodes -- so forming the matrix first
    would build 104 columns per row and throw almost all of them away.

    Differentiated with respect to the element's unknowns AND with respect to the nodal gradient,
    each contracted with its own direction. The second half is the part a lagged action drops, and
    it is not a small correction to the first: for a smooth field the reconstruction IS the
    correction, so dropping grad(v) drops the term's leading behaviour.

    ``sgn`` stays a symbol, so d|m|/dm = sgn exactly as the hand-written action assumes. That is a
    choice of subgradient at m = 0 and it is the same one the first-order kernel makes, which is
    what keeps the two demonstrably one linearisation.
    """
    r, mdots = residual_defcor_exprs(sym, rc=rc, limiter=limiter)
    pairs: list[tuple[sp.Symbol, sp.Symbol]] = []
    for a in range(N_NODE):
        pairs.append((sym["ux"][a], sym["vx"][a]))
        pairs.append((sym["uy"][a], sym["vy"][a]))
        pairs.append((sym["uz"][a], sym["vz"][a]))
        pairs.append((sym["p"][a], sym["q"][a]))
        for c in range(9):
            pairs.append((sym["g"][a][c], sym["gv"][a][c]))
    out = []
    for k in range(N_DOF):
        terms = []
        for var, dvar in pairs:
            if not r[k].has(var):
                continue
            d = sp.diff(r[k], var)
            if d != 0:
                terms.append(d * dvar)
        out.append(sp.Add(*terms) if terms else sp.Integer(0))
    return out, mdots


def simd_direction_locals_defcor() -> str:
    """The direction and its reconstructed gradient, hoisted per lane like the state is."""
    lines = []
    for name, field in (("vx", "ux"), ("vy", "uy"), ("vz", "uz"), ("q", "p")):
        for i in range(N_NODE):
            lines.append(f"        const scalar_t {name}{i} = dir.{field}[{i}][lane];")
    for a in range(N_NODE):
        for c in range(9):
            lines.append(f"        const scalar_t gv{a}_{c} = hov.g[{a}][{c}][lane];")
    return "\n".join(lines)


def defcor_action_kernel(sym: dict[str, object], limiter: int, rc: bool) -> str:
    """One lane-blocked EXACT higher-order Jacobian action: whole element, one CSE'd block."""
    action, mdots = defcor_action_exprs(sym, rc=rc, limiter=limiter)
    name = f"cvfem_hex8_ns_upwind_sympy_jv_defcor{'_rc' if rc else ''}_lim{limiter}_simd"
    rc_arg = "        const Hex8RhieChowPack &rc,\n" if rc else ""
    rc_loc = simd_rc_locals() + "\n" if rc else ""
    return f"""
// {LIMITER_NAME[limiter]}{', with Rhie-Chow' if rc else ''}, exact Jacobian action.
static SFEM_INLINE void {name}(
        const scalar_t rho,
        const scalar_t mu,
{defcor_simd_args()}
        const Hex8InputPack &in,
        const Hex8InputPack &dir,
        const Hex8UGradPack &ho,
        const Hex8UGradPack &hov,
{rc_arg}        Hex8ResidualPack    &out) {{
#pragma omp simd aligned(cof0_ptr, cof1_ptr, cof2_ptr, cof3_ptr, cof4_ptr, cof5_ptr, cof6_ptr, cof7_ptr, cof8_ptr, det_ptr : 64)
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {{
{simd_input_locals_defcor()}
{simd_direction_locals_defcor()}
{rc_loc}{sign_locals(mdots, indent="        ")}
{cse_code(action, residual_pack_outputs(), indent="        ")}
    }}
}}
"""


def defcor_action_kernels(sym: dict[str, object]) -> str:
    """The exact action for every limiter arm, with and without Rhie-Chow.

    Same eight-way split as the residual and for the same reason: a runtime limiter would put a
    four-way select inside the vector body at every surface and component.
    """
    return "".join(defcor_action_kernel(sym, lim, rc)
                   for rc in (False, True) for lim in (0, 1, 2, 3))


def direction_symbols(sym: dict[str, object]) -> list[sp.Expr]:
    """The direction, ordered to match the Jacobian's columns."""
    d = []
    for a in range(N_NODE):
        d.extend((sym["vx"][a], sym["vy"][a], sym["vz"][a], sym["q"][a]))
    return d


def action_exprs(jac: list[sp.Expr], sym: dict[str, object]) -> list[sp.Expr]:
    """The Jacobian action, J(u) v, as N_DOF expressions.

    This is the directional derivative of the residual, which is the Jacobian contracted
    with the direction. Building it from ``jac`` rather than by substituting u -> u + t v
    and differentiating in t costs nothing extra -- ``jac`` is already built for the
    assembly -- and it keeps the two kernels demonstrably the same operator.

    Why this exists at all: HEX8 has had no generated Jacobian action. The four `sympy*`
    kernel names cover the residual and the assembly only, and no `apply_jacobian_action_*`
    takes a kernel selector, so every CSE arrangement is unmeasured for the operation a
    Krylov solve spends its time in. TET4 has had one since it was written.
    """
    v = direction_symbols(sym)
    n = len(v)
    out = []
    for i in range(N_DOF):
        row = jac[i * n:(i + 1) * n]
        out.append(sp.Add(*[c * vj for c, vj in zip(row, v) if c != 0]))
    return out


def action_outputs() -> list[str]:
    return [f"r[{i}]" for i in range(N_DOF)]


def direction_locals() -> str:
    lines = []
    for name in ("vx", "vy", "vz"):
        for i in range(N_NODE):
            lines.append(f"    const scalar_t {name}{i} = {name}[{i}];")
    for i in range(N_NODE):
        lines.append(f"    const scalar_t q{i} = q[{i}];")
    return "\n".join(lines)


def hoist_geometry(exprs: list[sp.Expr], sym: dict[str, object]) -> tuple[list[sp.Expr], list[tuple[sp.Symbol, sp.Expr]]]:
    """Pull the maximal geometry-only subexpressions out into their own symbols.

    On an affine element all twelve sub-control surfaces share one adjugate, and the
    generator's own note says that is why the affine SymPy kernels beat the hand-written
    ones: CSE has a great deal to factor out. But that sharing is left for `sp.cse` to
    DISCOVER, across an expression set of a thousand terms in which the geometry is
    tangled with the fields. This makes it explicit instead -- every subtree whose free
    symbols are drawn only from cof0..8 and det becomes a `g` symbol, those are factored
    in their own pass, and the field algebra is then CSE'd with the geometry already
    reduced to atoms.

    Maximal, not every: the scan stops descending as soon as a subtree qualifies, so a
    geometry product is hoisted whole rather than as its factors.
    """
    geom = set(sym["cof"]) | {sym["det"]}
    subs: dict[sp.Expr, sp.Symbol] = {}
    gen = sp.numbered_symbols("g")

    def scan(e: sp.Expr) -> sp.Expr:
        if e.is_Atom:
            return e
        fs = e.free_symbols
        # `fs` non-empty excludes pure rationals, which are cheaper inline than as a name.
        if fs and fs <= geom:
            if e not in subs:
                subs[e] = next(gen)
            return subs[e]
        return e.func(*[scan(a) for a in e.args])

    out = [scan(e) for e in exprs]
    # Definition order is the order the symbols were created, so a later definition can
    # never reference an earlier one it has not seen -- they are disjoint subtrees.
    defs = sorted(subs.items(), key=lambda kv: int(str(kv[1])[1:]))
    return out, [(v, k) for k, v in defs]


def cse_action_geom_code(action: list[sp.Expr], sym: dict[str, object], facewise_jacs=None) -> str:
    """The action with the geometry hoisted into a first CSE pass.

    Two levels: the geometry-only subtrees are factored among themselves and emitted as
    `g` locals, then the field-dependent remainder is factored with those as atoms. The
    second level is either flat or face-wise, so the hoist can be measured both on its own
    and on top of the arrangement that already won.
    """
    if facewise_jacs is None:
        hoisted, defs = hoist_geometry(action, sym)
        body = [cse_emit([e for _v, e in defs], [str(v) for v, _e in defs],
                         mode="declare", prefix="gx")]
        body.append(cse_code(hoisted, action_outputs(), op="+="))
        return "\n".join(body)

    v = direction_symbols(sym)
    n = len(v)
    per_face = []
    for fj in facewise_jacs:
        exprs, outs = [], []
        for i in range(N_DOF):
            row = fj[i * n:(i + 1) * n]
            e = sp.Add(*[c * vj for c, vj in zip(row, v) if c != 0])
            if e != 0:
                exprs.append(e)
                outs.append(f"r[{i}]")
        per_face.append((exprs, outs))
    # One geometry pass for the whole kernel, not one per face: the faces share the
    # adjugate, which is the entire premise of hoisting it.
    flat = [e for exprs, _o in per_face for e in exprs]
    hoisted, defs = hoist_geometry(flat, sym)
    body = [cse_emit([e for _v, e in defs], [str(v_) for v_, _e in defs],
                     mode="declare", prefix="gx")]
    k = 0
    for exprs, outs in per_face:
        if not exprs:
            continue
        chunk = hoisted[k:k + len(exprs)]
        k += len(exprs)
        body.append("    {")
        body.append(cse_code(chunk, outs, indent="        ", op="+="))
        body.append("    }")
    return "\n".join(body)


def cse_action_facewise_code(face_jacs: list[list[sp.Expr]], sym: dict[str, object]) -> str:
    """The action accumulated one sub-control surface at a time.

    The finest cut available, and the one the other arrangements argue for: cutting the
    CSE scope beat the flat whole-kernel scope on this operator, so the question is how
    far that goes. Each face touches two nodes, so a scope here is eight outputs of much
    simpler algebra than any slice of the assembled twelve-face sum.

    Face-wise lost badly as an ASSEMBLY arrangement -- 24.0 against 54.3 MDOF/s on CPU
    atomic -- but for a reason that does not exist here: it issued 2016 CVFEM_ATOMIC_ADDs
    against flat's 768. The action accumulates into a local r[], so the extra writes are
    register or stack traffic rather than atomics, and that verdict does not carry over.
    """
    v = direction_symbols(sym)
    n = len(v)
    body = []
    for fj in face_jacs:
        exprs, outs = [], []
        for i in range(N_DOF):
            row = fj[i * n:(i + 1) * n]
            e = sp.Add(*[c * vj for c, vj in zip(row, v) if c != 0])
            if e != 0:
                exprs.append(e)
                outs.append(f"r[{i}]")
        if not exprs:
            continue
        body.append("    {")
        body.append(cse_code(exprs, outs, indent="        ", op="+="))
        body.append("    }")
    return "\n".join(body)


def cse_action_code(action: list[sp.Expr], scope: str) -> str:
    """Emit the action under one CSE arrangement.

    ``scope`` is the only thing that distinguishes the arrangements, because an
    arrangement IS the set of expressions handed to one sp.cse call:

    ``flat``       all N_DOF outputs in one scope -- maximum reuse, longest live ranges
    ``node``       the four dofs of one node per scope -- 8 scopes
    ``component``  one component across all nodes per scope -- 4 scopes, and the grouping
                   the momentum/continuity split suggests
    """
    outs = action_outputs()
    if scope == "flat":
        return cse_code(action, outs, op="+=")
    body = []
    if scope == "node":
        groups = [(a, list(range(a * 4, a * 4 + 4))) for a in range(N_NODE)]
    elif scope == "component":
        groups = [(c, list(range(c, N_DOF, 4))) for c in range(4)]
    else:
        raise ValueError("unknown action CSE scope: %s" % scope)
    for _tag, idx in groups:
        exprs = [action[i] for i in idx]
        if all(e == 0 for e in exprs):
            continue
        body.append("    {")
        body.append(cse_code(exprs, [outs[i] for i in idx], indent="        ", op="+="))
        body.append("    }")
    return "\n".join(body)


def cse_code(exprs: list[sp.Expr], outputs: list[str], indent: str = "    ", op: str = "=") -> str:
    return cse_emit(exprs, outputs, indent, mode="assign", op=op, drop_zeros=True)


def cse_atomic_add_code(exprs: list[sp.Expr], outputs: list[str], indent: str = "    ") -> str:
    return cse_emit(exprs, outputs, indent, mode="atomic_add", drop_zeros=True)


def input_locals(include_pressure: bool) -> str:
    lines = []
    for name in ("ux", "uy", "uz"):
        for i in range(N_NODE):
            lines.append(f"    const scalar_t {name}{i} = {name}[{i}];")
    if include_pressure:
        for i in range(N_NODE):
            lines.append(f"    const scalar_t p{i} = p[{i}];")
    return "\n".join(lines)


def geom_locals() -> str:
    lines = [f"    const scalar_t cof{i} = adj[{i}];" for i in range(9)]
    return "\n".join(lines)


def geom_locals_isoparam() -> str:
    """Evaluate the twelve sub-control-surface geometries, then name their components.

    This part is deliberately not generated symbolically. Expressing the adjugate as a
    polynomial in the 24 nodal coordinates and letting it feed the whole element matrix
    makes the expressions explode; calling cvfem_hex8_geom_at twelve times is what the
    hand-written isoparametric kernel does, costs the same arithmetic, and keeps the
    generated body to the algebra that CSE can actually improve.
    """
    lines = [
        "    scalar_t adjs[CVFEM_HEX8_N_SCS][9], dets[CVFEM_HEX8_N_SCS];",
        "    for (int s_ = 0; s_ < CVFEM_HEX8_N_SCS; ++s_) {",
        "        cvfem_hex8_geom_at<scalar_t>(x, y, z, CVFEM_HEX8_SCS_XI[s_][0], "
        "CVFEM_HEX8_SCS_XI[s_][1], CVFEM_HEX8_SCS_XI[s_][2], adjs[s_], &dets[s_]);",
        "    }",
    ]
    for s in range(12):
        for i in range(9):
            lines.append(f"    const scalar_t c{s}_{i} = adjs[{s}][{i}];")
        lines.append(f"    const scalar_t d{s} = dets[{s}];")
    return "\n".join(lines)


def sign_locals(mdots: list[sp.Expr], indent: str = "    ") -> str:
    # `indent` is forwarded rather than reimplemented: the lane-blocked kernel below sits one
    # level deeper than the scalar ones, and the shared emitter already takes it.
    return _sign_locals(mdots, indent)


def residual_outputs() -> list[str]:
    return [f"r[{i}]" for i in range(N_DOF)]


def jac_block_exprs(jac: list[sp.Expr], row_node: int, col_node: int) -> list[sp.Expr]:
    return _jac_block_exprs(jac, row_node, col_node, N_DOF)


def cse_add_bsr_slots_code(jac: list[sp.Expr], block_scope: str, atomic: bool) -> str:
    lines: list[str] = []
    if block_scope == "flat":
        outputs = []
        exprs = []
        for rn in range(N_NODE):
            for rf in range(N_FIELD):
                row = dof(rn, rf)
                for cn in range(N_NODE):
                    block = rn * N_NODE + cn
                    for cf in range(N_FIELD):
                        col = dof(cn, cf)
                        exprs.append(jac[row * N_DOF + col])
                        outputs.append(f"values[(ptrdiff_t)slots[{block}] * 16 + {rf * N_FIELD + cf}]")
        return cse_atomic_add_code(exprs, outputs) if atomic else cse_code(exprs, outputs, op="+=")

    for rn in range(N_NODE):
        for cn in range(N_NODE):
            block = rn * N_NODE + cn
            raw = jac_block_exprs(jac, rn, cn)
            if block_scope == "block":
                if not any(expr != 0 for expr in raw):
                    continue
                outputs = [f"block{block}[{i}]" for i in range(N_FIELD * N_FIELD)]
                lines.append("    {")
                lines.append(f"        scalar_t *const SFEM_RESTRICT block{block} = values + (ptrdiff_t)slots[{block}] * 16;")
                code = cse_atomic_add_code(raw, outputs, indent="        ") if atomic else cse_code(raw, outputs, indent="        ", op="+=")
                if code:
                    lines.append(code)
                lines.append("    }")
            else:
                for rf in range(N_FIELD):
                    row_exprs = raw[rf * N_FIELD:(rf + 1) * N_FIELD]
                    if not any(expr != 0 for expr in row_exprs):
                        continue
                    outputs = [f"row[{cf}]" for cf in range(N_FIELD)]
                    lines.append("    {")
                    lines.append(f"        scalar_t *const SFEM_RESTRICT row = values + (ptrdiff_t)slots[{block}] * 16 + {rf * N_FIELD};")
                    code = cse_atomic_add_code(row_exprs, outputs, indent="        ") if atomic else cse_code(row_exprs, outputs, indent="        ", op="+=")
                    if code:
                        lines.append(code)
                    lines.append("    }")
    return "\n".join(lines)


def cse_add_facewise_bsr_slots_code(face_jacs: list[list[sp.Expr]], atomic: bool) -> str:
    lines: list[str] = []
    for face, jac in enumerate(face_jacs):
        outputs = []
        exprs = []
        for rn in range(N_NODE):
            for rf in range(N_FIELD):
                row = dof(rn, rf)
                for cn in range(N_NODE):
                    block = rn * N_NODE + cn
                    for cf in range(N_FIELD):
                        col = dof(cn, cf)
                        exprs.append(jac[row * N_DOF + col])
                        outputs.append(f"values[(ptrdiff_t)slots[{block}] * 16 + {rf * N_FIELD + cf}]")
        code = cse_atomic_add_code(exprs, outputs, indent="        ") if atomic else cse_code(exprs, outputs, indent="        ", op="+=")
        if code:
            lines.append("    {")
            lines.append(f"        // SCS face {face}")
            lines.append(code)
            lines.append("    }")
    return "\n".join(lines)


def generate() -> str:
    sym = build_symbols()
    residual, mdots = residual_exprs(sym)
    q = []
    for a in range(N_NODE):
        q.extend((sym["ux"][a], sym["uy"][a], sym["uz"][a], sym["p"][a]))
    jac = [sp.diff(row, col) for row in residual for col in q]
    face_jacs = []
    for s in range(len(SCS)):
        face_residual, _mdot = face_residual_expr(sym, s)
        face_jacs.append([sp.diff(row, col) for row in face_residual for col in q])

    # Isoparametric: the same expressions with per-face geometry. Every face carries its
    # own adjugate, determinant and shape derivatives, so nothing is shared between
    # faces and CSE has far less to exploit than in the affine case -- which is the
    # substance of the comparison these kernels exist to make.
    iso_residual, iso_mdots = residual_exprs(sym, isoparam=True)
    iso_jac = [sp.diff(row, col) for row in iso_residual for col in q]

    # The Jacobian action, which HEX8 has never had in generated form. Three CSE
    # arrangements over the same expressions, so the scope question can finally be asked
    # of the operation a Krylov solve actually spends its time in.
    action = action_exprs(jac, sym)

    # The residual WITH the deferred correction, as one expression set for the whole element.
    # Generated rather than hand-written because the hand-written form hit a wall that is
    # structural rather than incidental: the reconstruction is a function, two compilers both
    # declined to inline it into the `#pragma omp simd` lane loop despite always_inline, and a
    # loop containing a call does not vectorise. There is no function here to inline. It also
    # lets CSE span all twelve surfaces, which twelve separately compiled per-face kernels
    # cannot do -- the eight node coordinates feeding the twelve centroids are shared once.
    # All four limiter arms, with and without Rhie-Chow. Separate kernels rather than one taking a
    # runtime limiter or a zero coefficient: either would put sweep-uniform work or a four-way select
    # inside the vector body at every surface, which is the shape of guard measured at 1.83x here.
    defcor_kernels_text = defcor_kernels(sym)

    return f"""#ifndef CVFEM_HEX8_NS_UPWIND_SYMPY_KERNELS_HPP
#define CVFEM_HEX8_NS_UPWIND_SYMPY_KERNELS_HPP

// Generated by synthesize_cvfem_hex8_ns_upwind_sympy.py. Do not edit by hand.
// SymPy {sp.__version__}. The CSE output is version-sensitive, so record the
// version that produced this file: regenerating under a different SymPy may
// legitimately reorder or rename temporaries.
//
// Not self-contained: the includer must already provide SFEM_RESTRICT and
// CVFEM_HEX8_N_DOF. Accumulation goes through CVFEM_ATOMIC_ADD.
#include "core/cvfem_portability.hpp"

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_residual(const scalar_t rho,
                                                            const scalar_t mu,
                                                            const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            const scalar_t *const SFEM_RESTRICT uz,
                                                            const scalar_t *const SFEM_RESTRICT p,
                                                            scalar_t *const SFEM_RESTRICT r) {{
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);
{geom_locals()}
{input_locals(include_pressure=True)}
{sign_locals(mdots)}
{cse_code(residual, residual_outputs(), op="+=")}
}}

// ---------------------------------------------------------------- Jacobian action
//
// J(u) v, generated. The three differ only in the scope each sp.cse call was given:
// `flat` sees all 32 outputs at once and has the most to reuse; `node` sees the four dofs
// of one node; `component` sees one component across all eight nodes. Which wins is a
// question about live ranges against reuse, and it is not answerable by inspection.
//
// The signs are inputs here exactly as they are in the residual and the assembly: the
// upwind switch is evaluated by the caller and enters as sgn0..sgn11, which is what makes
// the flux algebra differentiable and this kernel generatable at all.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_action(const scalar_t rho,
                                                            const scalar_t mu,
                                                            const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            const scalar_t *const SFEM_RESTRICT uz,
                                                            const scalar_t *const SFEM_RESTRICT vx,
                                                            const scalar_t *const SFEM_RESTRICT vy,
                                                            const scalar_t *const SFEM_RESTRICT vz,
                                                            const scalar_t *const SFEM_RESTRICT q,
                                                            scalar_t *const SFEM_RESTRICT r) {{
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);
{geom_locals()}
{input_locals(include_pressure=False)}
{direction_locals()}
{sign_locals(mdots)}
{cse_action_code(action, "flat")}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_action_nodewise(const scalar_t rho,
                                                            const scalar_t mu,
                                                            const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            const scalar_t *const SFEM_RESTRICT uz,
                                                            const scalar_t *const SFEM_RESTRICT vx,
                                                            const scalar_t *const SFEM_RESTRICT vy,
                                                            const scalar_t *const SFEM_RESTRICT vz,
                                                            const scalar_t *const SFEM_RESTRICT q,
                                                            scalar_t *const SFEM_RESTRICT r) {{
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);
{geom_locals()}
{input_locals(include_pressure=False)}
{direction_locals()}
{sign_locals(mdots)}
{cse_action_code(action, "node")}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_action_componentwise(const scalar_t rho,
                                                            const scalar_t mu,
                                                            const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            const scalar_t *const SFEM_RESTRICT uz,
                                                            const scalar_t *const SFEM_RESTRICT vx,
                                                            const scalar_t *const SFEM_RESTRICT vy,
                                                            const scalar_t *const SFEM_RESTRICT vz,
                                                            const scalar_t *const SFEM_RESTRICT q,
                                                            scalar_t *const SFEM_RESTRICT r) {{
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);
{geom_locals()}
{input_locals(include_pressure=False)}
{direction_locals()}
{sign_locals(mdots)}
{cse_action_code(action, "component")}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_action_facewise(const scalar_t rho,
                                                            const scalar_t mu,
                                                            const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            const scalar_t *const SFEM_RESTRICT uz,
                                                            const scalar_t *const SFEM_RESTRICT vx,
                                                            const scalar_t *const SFEM_RESTRICT vy,
                                                            const scalar_t *const SFEM_RESTRICT vz,
                                                            const scalar_t *const SFEM_RESTRICT q,
                                                            scalar_t *const SFEM_RESTRICT r) {{
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);
{geom_locals()}
{input_locals(include_pressure=False)}
{direction_locals()}
{sign_locals(mdots)}
{cse_action_facewise_code(face_jacs, sym)}
}}

// Two-level: the geometry factored in its own pass, then the field algebra with the
// geometry already reduced to `g` atoms. Emitted twice, flat and face-wise, so the hoist
// can be read both on its own and on top of the arrangement that already won -- which is
// the only way to tell "the hoist helps" from "the hoist helps where nothing else did".
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_action_geom(const scalar_t rho,
                                                            const scalar_t mu,
                                                            const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            const scalar_t *const SFEM_RESTRICT uz,
                                                            const scalar_t *const SFEM_RESTRICT vx,
                                                            const scalar_t *const SFEM_RESTRICT vy,
                                                            const scalar_t *const SFEM_RESTRICT vz,
                                                            const scalar_t *const SFEM_RESTRICT q,
                                                            scalar_t *const SFEM_RESTRICT r) {{
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);
{geom_locals()}
{input_locals(include_pressure=False)}
{direction_locals()}
{sign_locals(mdots)}
{cse_action_geom_code(action, sym)}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_action_geomface(const scalar_t rho,
                                                            const scalar_t mu,
                                                            const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            const scalar_t *const SFEM_RESTRICT uz,
                                                            const scalar_t *const SFEM_RESTRICT vx,
                                                            const scalar_t *const SFEM_RESTRICT vy,
                                                            const scalar_t *const SFEM_RESTRICT vz,
                                                            const scalar_t *const SFEM_RESTRICT q,
                                                            scalar_t *const SFEM_RESTRICT r) {{
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);
{geom_locals()}
{input_locals(include_pressure=False)}
{direction_locals()}
{sign_locals(mdots)}
{cse_action_geom_code(action, sym, facewise_jacs=face_jacs)}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots(const scalar_t rho,
                                                                          const scalar_t mu,
                                                                          const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                                          const scalar_t *const SFEM_RESTRICT ux,
                                                                          const scalar_t *const SFEM_RESTRICT uy,
                                                                          const scalar_t *const SFEM_RESTRICT uz,
                                                                          const smesh::count_t *const SFEM_RESTRICT slots,
                                                                          scalar_t *const SFEM_RESTRICT values) {{
{geom_locals()}
{input_locals(include_pressure=False)}
{sign_locals(mdots)}
{cse_add_bsr_slots_code(jac, "flat", atomic=True)}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_blockwise(const scalar_t rho,
                                                                                    const scalar_t mu,
                                                                                    const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                                                    const scalar_t *const SFEM_RESTRICT ux,
                                                                                    const scalar_t *const SFEM_RESTRICT uy,
                                                                                    const scalar_t *const SFEM_RESTRICT uz,
                                                                                    const smesh::count_t *const SFEM_RESTRICT slots,
                                                                                    scalar_t *const SFEM_RESTRICT values) {{
{geom_locals()}
{input_locals(include_pressure=False)}
{sign_locals(mdots)}
{cse_add_bsr_slots_code(jac, "block", atomic=True)}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise(const scalar_t rho,
                                                                                  const scalar_t mu,
                                                                                  const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                                                  const scalar_t *const SFEM_RESTRICT ux,
                                                                                  const scalar_t *const SFEM_RESTRICT uy,
                                                                                  const scalar_t *const SFEM_RESTRICT uz,
                                                                                  const smesh::count_t *const SFEM_RESTRICT slots,
                                                                                  scalar_t *const SFEM_RESTRICT values) {{
{geom_locals()}
{input_locals(include_pressure=False)}
{sign_locals(mdots)}
{cse_add_bsr_slots_code(jac, "row", atomic=True)}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_facewise(const scalar_t rho,
                                                                                   const scalar_t mu,
                                                                                   const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                                                   const scalar_t *const SFEM_RESTRICT ux,
                                                                                   const scalar_t *const SFEM_RESTRICT uy,
                                                                                   const scalar_t *const SFEM_RESTRICT uz,
                                                                                   const smesh::count_t *const SFEM_RESTRICT slots,
                                                                                   scalar_t *const SFEM_RESTRICT values) {{
{geom_locals()}
{input_locals(include_pressure=False)}
{sign_locals(mdots)}
{cse_add_facewise_bsr_slots_code(face_jacs, atomic=True)}
}}

// ---------------------------------------------------------------- isoparametric
//
// Same algebra, per-face geometry. The twelve sub-control-surface Jacobians are
// evaluated by ordinary code rather than generated: expressing the adjugate as a
// polynomial in the 24 nodal coordinates and letting it feed the element matrix makes
// the expressions explode, and it would not save any arithmetic, since the hand-written
// kernel evaluates exactly the same twelve geometries.
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_residual_isoparam(const scalar_t rho,
                                                            const scalar_t mu,
                                                            const scalar_t *const SFEM_RESTRICT x,
                                                            const scalar_t *const SFEM_RESTRICT y,
                                                            const scalar_t *const SFEM_RESTRICT z,
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            const scalar_t *const SFEM_RESTRICT uz,
                                                            const scalar_t *const SFEM_RESTRICT p,
                                                            scalar_t *const SFEM_RESTRICT r) {{
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);
{geom_locals_isoparam()}
{input_locals(include_pressure=True)}
{sign_locals(iso_mdots)}
{cse_code(iso_residual, residual_outputs(), op="+=")}
}}

template <typename scalar_t, typename Slot>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_isoparam(const scalar_t rho,
                                                                                   const scalar_t mu,
                                                                                   const scalar_t *const SFEM_RESTRICT x,
                                                                                   const scalar_t *const SFEM_RESTRICT y,
                                                                                   const scalar_t *const SFEM_RESTRICT z,
                                                                                   const scalar_t *const SFEM_RESTRICT ux,
                                                                                   const scalar_t *const SFEM_RESTRICT uy,
                                                                                   const scalar_t *const SFEM_RESTRICT uz,
                                                                                   const Slot *const SFEM_RESTRICT slots,
                                                                                   scalar_t *const SFEM_RESTRICT values) {{
{geom_locals_isoparam()}
{input_locals(include_pressure=False)}
{sign_locals(iso_mdots)}
{cse_add_bsr_slots_code(iso_jac, "flat", atomic=True)}
}}

// ------------------------------------------------- the deferred correction, lane-blocked
//
// The complete element residual including the higher-order deferred correction, sixteen elements at
// a time, reading the lane-major packs in place. One kernel per limiter arm and per Rhie-Chow state.
//
// Each is the whole element as ONE common-subexpression-eliminated block inside ONE lane loop, and
// both of those matter. One block, because the eight node coordinates feed all twelve sub-control-
// surface centroids and CSE can only see that if the twelve surfaces are emitted together -- a
// per-face kernel, compiled twelve times, reloads them twelve times. One loop with no calls in it,
// because the hand-written equivalent could not keep the reconstruction inlined: clang emitted an
// out-of-line copy and gcc a .constprop clone and both CALLED it from inside the `#pragma omp simd`
// body, which stops the loop vectorising outright. There is no function here for a compiler to
// decline to inline.
//
// The upwind split is written once per surface by the flux and again by the correction; CSE unifies
// them, so the emitted text carries it once.
{defcor_kernels_text}
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots(const scalar_t rho,
                                                                            const scalar_t mu,
                                                                            const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                                            const scalar_t *const SFEM_RESTRICT uz,
                                                                            const int *const SFEM_RESTRICT slots,
                                                                            scalar_t *const SFEM_RESTRICT values) {{
{geom_locals()}
{input_locals(include_pressure=False)}
{sign_locals(mdots)}
{cse_add_bsr_slots_code(jac, "flat", atomic=False)}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_blockwise(const scalar_t rho,
                                                                                      const scalar_t mu,
                                                                                      const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                                                      const scalar_t *const SFEM_RESTRICT ux,
                                                                                      const scalar_t *const SFEM_RESTRICT uy,
                                                                                      const scalar_t *const SFEM_RESTRICT uz,
                                                                                      const int *const SFEM_RESTRICT slots,
                                                                                      scalar_t *const SFEM_RESTRICT values) {{
{geom_locals()}
{input_locals(include_pressure=False)}
{sign_locals(mdots)}
{cse_add_bsr_slots_code(jac, "block", atomic=False)}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_rowwise(const scalar_t rho,
                                                                                    const scalar_t mu,
                                                                                    const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                                                    const scalar_t *const SFEM_RESTRICT ux,
                                                                                    const scalar_t *const SFEM_RESTRICT uy,
                                                                                    const scalar_t *const SFEM_RESTRICT uz,
                                                                                    const int *const SFEM_RESTRICT slots,
                                                                                    scalar_t *const SFEM_RESTRICT values) {{
{geom_locals()}
{input_locals(include_pressure=False)}
{sign_locals(mdots)}
{cse_add_bsr_slots_code(jac, "row", atomic=False)}
}}

template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_facewise(const scalar_t rho,
                                                                                     const scalar_t mu,
                                                                                     const scalar_t *const SFEM_RESTRICT adj, const scalar_t det,
                                                                                     const scalar_t *const SFEM_RESTRICT ux,
                                                                                     const scalar_t *const SFEM_RESTRICT uy,
                                                                                     const scalar_t *const SFEM_RESTRICT uz,
                                                                                     const int *const SFEM_RESTRICT slots,
                                                                                     scalar_t *const SFEM_RESTRICT values) {{
{geom_locals()}
{input_locals(include_pressure=False)}
{sign_locals(mdots)}
{cse_add_facewise_bsr_slots_code(face_jacs, atomic=False)}
}}

#endif
"""


# CSE arrangements that lost the saturated evaluation and moved to subpar/. They are
# still generated -- the removal was made on measured grounds and has to stay
# reproducible -- but into a separate header that only builds under
# -DCVFEM_ENABLE_SUBPAR. See subpar/README.md for the numbers.
# The quarantined ARRANGEMENTS, named precisely rather than by substring.
#
# This was ("_rowwise", "_facewise"), which quarantines any function whose name happens to
# contain either -- and it silently swallowed the first new kernel to use one of those
# words, a Jacobian-action arrangement that has never been measured and that the assembly
# verdict says nothing about. A rule that decides by substring cannot distinguish a variant
# that lost from one that merely shares a word with it.
SUBPAR_MARKERS = (
                  "cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise",
                  "cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_facewise",
                  "cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_rowwise",
                  "cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_facewise",
                  # The six generated Jacobian-action arrangements, measured on Grace at
                  # 8,586,756 dof against the hand-written atomic action's 828.9 MDOF/s
                  # (perf/campaign_generated_arms.csv, 3 reps):
                  #
                  #   action           307.8   0.37x      action_geom      359.2   0.43x
                  #   action_comp      328.2   0.40x      action_face      418.6   0.51x
                  #   action_node      329.3   0.40x      action_geomface  473.3   0.57x
                  #
                  # The best of the six reaches 57% of the hand-written atomic action and 25%
                  # of the packed one (1905.3). The axis they explore -- the scope given to one
                  # sp.cse call -- is genuinely interesting and the spread across it is 1.5x,
                  # which is why they were built; none of it closes a gap this size.
                  "cvfem_hex8_ns_upwind_sympy_jacobian_action",
                  "cvfem_hex8_ns_upwind_sympy_jacobian_action_componentwise",
                  "cvfem_hex8_ns_upwind_sympy_jacobian_action_facewise",
                  "cvfem_hex8_ns_upwind_sympy_jacobian_action_geom",
                  "cvfem_hex8_ns_upwind_sympy_jacobian_action_geomface",
                  "cvfem_hex8_ns_upwind_sympy_jacobian_action_nodewise",
                  # The affine generated residual. NOT a flat loss and the entry in
                  # subpar/README.md says so: on the packed layout it gives up 21.6%
                  # (2066.9 against sumfact's 2636.8), on the atomic layout it TIES sumfact
                  # (854.8 against 858.2, inside the spread) and trails `current` by 3.9%.
                  # It is quarantined for never winning anywhere rather than for losing
                  # everywhere, which are different findings.
                  #
                  # _residual_isoparam is deliberately absent: it is the isoparametric scalar
                  # winner and this campaign measured affine geometry only, so there is no
                  # fresh evidence about it and it is not being retired on old evidence.
                  "cvfem_hex8_ns_upwind_sympy_residual")

SUBPAR_OUT = SPIKE_ROOT / "subpar" / "cvfem_hex8_ns_upwind_sympy_subpar.hpp"


def split_generated(text: str) -> tuple[str, str]:
    """Partition the generated header into survivors and quarantined arrangements.

    The functions are *moved*, not re-emitted: both outputs carry the exact text this
    run produced, so a survivor cannot drift as a side effect of the split. That is the
    property worth having -- the alternative, generating each set separately, would let
    a change in which expressions are built perturb the CSE of the ones that stayed.
    """
    marker = "template <typename scalar_t"
    first = text.index(marker)
    prologue, tail = text[:first], "\n#endif\n"
    body = text[first : text.rindex("#endif")]

    chunks, keep, drop = [], [], []
    idx = [i for i in range(len(body)) if body.startswith(marker, i)]
    for a, b in zip(idx, idx[1:] + [len(body)]):
        chunks.append(body[a:b])
    # EXACT function names, not substrings. The previous rule asked whether a marker was a
    # substring of the text before the first "(", which cannot tell a name from that same name
    # with a suffix: quarantining `_residual` would silently take `_residual_isoparam` with it,
    # and that one is a measured WINNER. The earlier incarnation of this rule already swallowed
    # an unmeasured kernel for the same reason, which is why the comment above says to name
    # them precisely -- this makes the code enforce what the comment asks for.
    def fn_name(chunk: str) -> str:
        head = chunk.split("(")[0]
        m = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", head)
        return m[-1] if m else ""

    for c in chunks:
        (drop if fn_name(c) in SUBPAR_MARKERS else keep).append(c)

    subpar_prologue = prologue.replace(
        "CVFEM_HEX8_NS_UPWIND_SYMPY_KERNELS_HPP", "CVFEM_HEX8_NS_UPWIND_SYMPY_SUBPAR_HPP"
    ).replace(
        "// Not self-contained:",
        "// QUARANTINED. These CSE arrangements are not the fastest choice in any measured\n"
        "// configuration on either platform; see subpar/README.md. Built only under\n"
        "// -DCVFEM_ENABLE_SUBPAR.\n"
        "//\n"
        "// Not self-contained:",
    )
    return prologue + "".join(keep) + tail, subpar_prologue + "".join(drop) + tail


def main() -> int:
    args = add_output_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    main_hpp, subpar_hpp = split_generated(generate())
    out = args.out or OUT
    # --out relocates the main header; the quarantined half follows it, so a check
    # against a scratch copy still exercises both.
    subpar_out = SUBPAR_OUT if args.out is None else out.parent / SUBPAR_OUT.name
    return emit([(out, main_hpp), (subpar_out, subpar_hpp)], args.check)


if __name__ == "__main__":
    raise SystemExit(main())
