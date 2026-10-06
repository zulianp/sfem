#!/usr/bin/env python3
"""Generate SymPy/CSE CVFEM HEX8 Navier-Stokes kernels."""

from __future__ import annotations

import argparse
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
OUT = SPIKE_ROOT / "src" / "generated" / "cvfem_hex8_ns_upwind_sympy_kernels.hpp"

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


def sign_locals(mdots: list[sp.Expr]) -> str:
    return _sign_locals(mdots)


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

    return f"""#ifndef CVFEM_HEX8_NS_UPWIND_SYMPY_KERNELS_HPP
#define CVFEM_HEX8_NS_UPWIND_SYMPY_KERNELS_HPP

// Generated by synthesize_cvfem_hex8_ns_upwind_sympy.py. Do not edit by hand.
// SymPy {sp.__version__}. The CSE output is version-sensitive, so record the
// version that produced this file: regenerating under a different SymPy may
// legitimately reorder or rename temporaries.
//
// Not self-contained: the includer must already provide SFEM_RESTRICT and
// CVFEM_HEX8_N_DOF. Accumulation goes through CVFEM_ATOMIC_ADD.
#include "cvfem_portability.hpp"

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
SUBPAR_MARKERS = ("_add_bsr_slots_rowwise", "_add_bsr_slots_facewise",
                  "_add_local_slots_rowwise", "_add_local_slots_facewise")

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
    for c in chunks:
        (drop if any(m in c.split("(")[0] for m in SUBPAR_MARKERS) else keep).append(c)

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
