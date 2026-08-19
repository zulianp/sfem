#!/usr/bin/env python3
"""Generate SymPy/CSE CVFEM TET4 Navier-Stokes kernels.

The generated kernels are reference alternatives for the hand-written residual
and Jacobian microkernels in cvfem_tet4_ns_upwind_kernels.hpp.
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp
from sympy.printing.c import C99CodePrinter


HERE = Path(__file__).resolve().parent
OUT = HERE / "cvfem_tet4_ns_upwind_sympy_kernels.hpp"

N_NODE = 4
N_FIELD = 4
N_DOF = N_NODE * N_FIELD

SCS = (
    (0, 1, (sp.Rational(1, 12), sp.Rational(1, 24), sp.Rational(1, 24))),
    (0, 2, (sp.Rational(1, 24), sp.Rational(1, 12), sp.Rational(1, 24))),
    (0, 3, (sp.Rational(1, 24), sp.Rational(1, 24), sp.Rational(1, 12))),
    (1, 2, (-sp.Rational(1, 24), sp.Rational(1, 24), sp.Rational(0))),
    (1, 3, (-sp.Rational(1, 24), sp.Rational(0), sp.Rational(1, 24))),
    (2, 3, (sp.Rational(0), -sp.Rational(1, 24), sp.Rational(1, 24))),
)


class ScalarPrinter(C99CodePrinter):
    def _print_Rational(self, expr: sp.Rational) -> str:
        return f"scalar_t({expr.p}) / scalar_t({expr.q})"

    def _print_Integer(self, expr: sp.Integer) -> str:
        return f"scalar_t({int(expr)})"

    def _print_Float(self, expr: sp.Float) -> str:
        return f"scalar_t({float(expr):.17g})"


def dof(node: int, field: int) -> int:
    return node * N_FIELD + field


def build_symbols() -> dict[str, object]:
    return {
        "rho": sp.Symbol("rho"),
        "mu": sp.Symbol("mu"),
        "det": sp.Symbol("det"),
        "adj": sp.symbols("adj0:9"),
        "ux": sp.symbols("ux0:4"),
        "uy": sp.symbols("uy0:4"),
        "uz": sp.symbols("uz0:4"),
        "p": sp.symbols("p0:4"),
        "sgn": sp.symbols("sgn0:6"),
    }


def scs_area(adj: tuple[sp.Symbol, ...], area_ref: tuple[sp.Rational, sp.Rational, sp.Rational]) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    ar0, ar1, ar2 = area_ref
    return (
        adj[0] * ar0 + adj[3] * ar1 + adj[6] * ar2,
        adj[1] * ar0 + adj[4] * ar1 + adj[7] * ar2,
        adj[2] * ar0 + adj[5] * ar1 + adj[8] * ar2,
    )


def velocity_gradient(sym: dict[str, object]) -> tuple[sp.Expr, ...]:
    adj = sym["adj"]
    det = sym["det"]
    ux = sym["ux"]
    uy = sym["uy"]
    uz = sym["uz"]
    inv_det = 1 / det
    out: list[sp.Expr] = []
    for comp in (ux, uy, uz):
        d0 = comp[1] - comp[0]
        d1 = comp[2] - comp[0]
        d2 = comp[3] - comp[0]
        out.extend(
            (
                (d0 * adj[0] + d1 * adj[3] + d2 * adj[6]) * inv_det,
                (d0 * adj[1] + d1 * adj[4] + d2 * adj[7]) * inv_det,
                (d0 * adj[2] + d1 * adj[5] + d2 * adj[8]) * inv_det,
            )
        )
    return tuple(out)


def residual_exprs(sym: dict[str, object], semismooth_abs: bool) -> tuple[list[sp.Expr], list[sp.Expr]]:
    rho = sym["rho"]
    mu = sym["mu"]
    adj = sym["adj"]
    ux = sym["ux"]
    uy = sym["uy"]
    uz = sym["uz"]
    p = sym["p"]
    sgn = sym["sgn"]

    g00, g01, g02, g10, g11, g12, g20, g21, g22 = velocity_gradient(sym)
    r = [sp.Integer(0)] * N_DOF
    mdots: list[sp.Expr] = []

    for s, (i, j, area_ref) in enumerate(SCS):
        ax, ay, az = scs_area(adj, area_ref)
        adv_x = sp.Rational(1, 2) * (ux[i] + ux[j])
        adv_y = sp.Rational(1, 2) * (uy[i] + uy[j])
        adv_z = sp.Rational(1, 2) * (uz[i] + uz[j])
        mdot = rho * (adv_x * ax + adv_y * ay + adv_z * az)
        mdots.append(mdot)

        mdot_abs = sgn[s] * mdot if semismooth_abs else sp.Abs(mdot)
        mdot_pos = sp.Rational(1, 2) * (mdot + mdot_abs)
        mdot_neg = sp.Rational(1, 2) * (mdot - mdot_abs)
        p_mid = sp.Rational(1, 2) * (p[i] + p[j])

        tau_x = mu * ((2 * g00) * ax + (g01 + g10) * ay + (g02 + g20) * az)
        tau_y = mu * ((g10 + g01) * ax + (2 * g11) * ay + (g12 + g21) * az)
        tau_z = mu * ((g20 + g02) * ax + (g21 + g12) * ay + (2 * g22) * az)

        fx = mdot_pos * ux[i] + mdot_neg * ux[j] + p_mid * ax - tau_x
        fy = mdot_pos * uy[i] + mdot_neg * uy[j] + p_mid * ay - tau_y
        fz = mdot_pos * uz[i] + mdot_neg * uz[j] + p_mid * az - tau_z

        r[dof(i, 0)] += fx
        r[dof(i, 1)] += fy
        r[dof(i, 2)] += fz
        r[dof(i, 3)] += mdot
        r[dof(j, 0)] -= fx
        r[dof(j, 1)] -= fy
        r[dof(j, 2)] -= fz
        r[dof(j, 3)] -= mdot

    return r, mdots


def cse_code(exprs: list[sp.Expr], outputs: list[str], indent: str = "    ") -> str:
    printer = ScalarPrinter()
    replacements, reduced = sp.cse(exprs, symbols=sp.numbered_symbols("x"), optimizations="basic")
    lines: list[str] = []
    for var, expr in replacements:
        lines.append(f"{indent}const scalar_t {var} = {printer.doprint(expr)};")
    for out, expr in zip(outputs, reduced):
        lines.append(f"{indent}{out} = {printer.doprint(expr)};")
    return "\n".join(lines)


def input_locals(include_pressure: bool) -> str:
    lines: list[str] = []
    for name in ("ux", "uy", "uz"):
        for i in range(4):
            lines.append(f"    const scalar_t {name}{i} = {name}[{i}];")
    if include_pressure:
        for i in range(4):
            lines.append(f"    const scalar_t p{i} = p[{i}];")
    return "\n".join(lines)


def simd_input_locals() -> str:
    lines = [
        "        const scalar_t adj0 = scalar_t(adj0_ptr[lane]);",
        "        const scalar_t adj1 = scalar_t(adj1_ptr[lane]);",
        "        const scalar_t adj2 = scalar_t(adj2_ptr[lane]);",
        "        const scalar_t adj3 = scalar_t(adj3_ptr[lane]);",
        "        const scalar_t adj4 = scalar_t(adj4_ptr[lane]);",
        "        const scalar_t adj5 = scalar_t(adj5_ptr[lane]);",
        "        const scalar_t adj6 = scalar_t(adj6_ptr[lane]);",
        "        const scalar_t adj7 = scalar_t(adj7_ptr[lane]);",
        "        const scalar_t adj8 = scalar_t(adj8_ptr[lane]);",
        "        const scalar_t det = scalar_t(det_ptr[lane]);",
    ]
    for name in ("ux", "uy", "uz", "p"):
        for i in range(4):
            lines.append(f"        const scalar_t {name}{i} = in.{name}[{i}][lane];")
    return "\n".join(lines)


def sign_locals(mdots: list[sp.Expr]) -> str:
    printer = ScalarPrinter()
    lines: list[str] = []
    for s, mdot in enumerate(mdots):
        expr = printer.doprint(mdot)
        lines.append(f"    const scalar_t mdot{s} = {expr};")
        lines.append(
            f"    const scalar_t sgn{s} = mdot{s} > scalar_t(0) ? scalar_t(1) : "
            f"(mdot{s} < scalar_t(0) ? scalar_t(-1) : scalar_t(0));"
        )
    return "\n".join(lines)


def residual_pack_outputs() -> list[str]:
    names = ("rx", "ry", "rz", "rc")
    return [f"out.{names[field]}[{node}][lane]" for node in range(N_NODE) for field in range(N_FIELD)]


def generate() -> str:
    sym = build_symbols()
    residual, _ = residual_exprs(sym, semismooth_abs=False)
    jac_residual, mdots = residual_exprs(sym, semismooth_abs=True)
    q = []
    for a in range(4):
        q.extend((sym["ux"][a], sym["uy"][a], sym["uz"][a], sym["p"][a]))
    jac = [sp.diff(row, col) for row in jac_residual for col in q]

    residual_outputs = [f"r[{i}]" for i in range(N_DOF)]
    jac_outputs = [f"ke[{i}]" for i in range(N_DOF * N_DOF)]

    return f"""#ifndef CVFEM_TET4_NS_UPWIND_SYMPY_KERNELS_HPP
#define CVFEM_TET4_NS_UPWIND_SYMPY_KERNELS_HPP

#include <cmath>

// Generated by synthesize_cvfem_tet4_ns_upwind_sympy.py. Do not edit by hand.

static SFEM_INLINE void cvfem_tet4_ns_upwind_sympy_residual_dense(const scalar_t rho,
                                                                  const scalar_t mu,
                                                                  const scalar_t adj0,
                                                                  const scalar_t adj1,
                                                                  const scalar_t adj2,
                                                                  const scalar_t adj3,
                                                                  const scalar_t adj4,
                                                                  const scalar_t adj5,
                                                                  const scalar_t adj6,
                                                                  const scalar_t adj7,
                                                                  const scalar_t adj8,
                                                                  const scalar_t det,
                                                                  const scalar_t ux[4],
                                                                  const scalar_t uy[4],
                                                                  const scalar_t uz[4],
                                                                  const scalar_t p[4],
                                                                  scalar_t *const SFEM_RESTRICT r) {{
{input_locals(include_pressure=True)}
{cse_code(residual, residual_outputs)}
}}

static SFEM_INLINE void cvfem_tet4_ns_upwind_sympy_residual_simd_microkernel(
        const scalar_t                        rho,
        const scalar_t                        mu,
        const jacobian_t *const SFEM_RESTRICT adj0_ptr,
        const jacobian_t *const SFEM_RESTRICT adj1_ptr,
        const jacobian_t *const SFEM_RESTRICT adj2_ptr,
        const jacobian_t *const SFEM_RESTRICT adj3_ptr,
        const jacobian_t *const SFEM_RESTRICT adj4_ptr,
        const jacobian_t *const SFEM_RESTRICT adj5_ptr,
        const jacobian_t *const SFEM_RESTRICT adj6_ptr,
        const jacobian_t *const SFEM_RESTRICT adj7_ptr,
        const jacobian_t *const SFEM_RESTRICT adj8_ptr,
        const jacobian_t *const SFEM_RESTRICT det_ptr,
        const Tet4InputPack                  &in,
        Tet4ResidualPack                     &out) {{
#pragma omp simd aligned(adj0_ptr, adj1_ptr, adj2_ptr, adj3_ptr, adj4_ptr, adj5_ptr, adj6_ptr, adj7_ptr, adj8_ptr, det_ptr : 64)
    for (int lane = 0; lane < VEC_SIZE; ++lane) {{
{simd_input_locals()}
{cse_code(residual, residual_pack_outputs(), indent="        ")}
    }}
}}

static SFEM_INLINE void cvfem_run_residual_sympy_kernel(const scalar_t                        rho,
                                                        const scalar_t                        mu,
                                                        const jacobian_t *const SFEM_RESTRICT adj0,
                                                        const jacobian_t *const SFEM_RESTRICT adj1,
                                                        const jacobian_t *const SFEM_RESTRICT adj2,
                                                        const jacobian_t *const SFEM_RESTRICT adj3,
                                                        const jacobian_t *const SFEM_RESTRICT adj4,
                                                        const jacobian_t *const SFEM_RESTRICT adj5,
                                                        const jacobian_t *const SFEM_RESTRICT adj6,
                                                        const jacobian_t *const SFEM_RESTRICT adj7,
                                                        const jacobian_t *const SFEM_RESTRICT adj8,
                                                        const jacobian_t *const SFEM_RESTRICT det,
                                                        const int                             nlanes,
                                                        const Tet4InputPack                  &in,
                                                        Tet4ResidualPack                     &out) {{
    if (nlanes == VEC_SIZE) {{
        cvfem_tet4_ns_upwind_sympy_residual_simd_microkernel(rho,
                                                             mu,
                                                             cvfem_aligned_geom(adj0),
                                                             cvfem_aligned_geom(adj1),
                                                             cvfem_aligned_geom(adj2),
                                                             cvfem_aligned_geom(adj3),
                                                             cvfem_aligned_geom(adj4),
                                                             cvfem_aligned_geom(adj5),
                                                             cvfem_aligned_geom(adj6),
                                                             cvfem_aligned_geom(adj7),
                                                             cvfem_aligned_geom(adj8),
                                                             cvfem_aligned_geom(det),
                                                             in,
                                                             out);
        return;
    }}
    alignas(ALIGN_BYTES) jacobian_t a0[VEC_SIZE], a1[VEC_SIZE], a2[VEC_SIZE], a3[VEC_SIZE], a4[VEC_SIZE];
    alignas(ALIGN_BYTES) jacobian_t a5[VEC_SIZE], a6[VEC_SIZE], a7[VEC_SIZE], a8[VEC_SIZE], detp[VEC_SIZE];
    cvfem_pad_geom_lanes(adj0, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, det, nlanes, a0, a1, a2, a3, a4, a5, a6, a7, a8, detp);
    cvfem_tet4_ns_upwind_sympy_residual_simd_microkernel(rho, mu, a0, a1, a2, a3, a4, a5, a6, a7, a8, detp, in, out);
}}

static SFEM_INLINE void cvfem_tet4_ns_upwind_sympy_jacobian_dense(const scalar_t rho,
                                                                  const scalar_t mu,
                                                                  const scalar_t adj0,
                                                                  const scalar_t adj1,
                                                                  const scalar_t adj2,
                                                                  const scalar_t adj3,
                                                                  const scalar_t adj4,
                                                                  const scalar_t adj5,
                                                                  const scalar_t adj6,
                                                                  const scalar_t adj7,
                                                                  const scalar_t adj8,
                                                                  const scalar_t det,
                                                                  const scalar_t ux[4],
                                                                  const scalar_t uy[4],
                                                                  const scalar_t uz[4],
                                                                  scalar_t *const SFEM_RESTRICT ke) {{
{input_locals(include_pressure=False)}
{sign_locals(mdots)}
{cse_code(jac, jac_outputs)}
}}

#endif
"""


def main() -> None:
    OUT.write_text(generate())


if __name__ == "__main__":
    main()
