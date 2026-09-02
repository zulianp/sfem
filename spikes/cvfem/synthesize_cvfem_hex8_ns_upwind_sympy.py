#!/usr/bin/env python3
"""Generate SymPy/CSE CVFEM HEX8 Navier-Stokes kernels."""

from __future__ import annotations

from pathlib import Path

import sympy as sp
from sympy.printing.c import C99CodePrinter


HERE = Path(__file__).resolve().parent
OUT = HERE / "cvfem_hex8_ns_upwind_sympy_kernels.hpp"

N_NODE = 8
N_FIELD = 4
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


class ScalarPrinter(C99CodePrinter):
    def _print_Rational(self, expr: sp.Rational) -> str:
        return f"scalar_t({expr.p}) / scalar_t({expr.q})"

    def _print_Integer(self, expr: sp.Integer) -> str:
        return f"scalar_t({int(expr)})"

    def _print_Float(self, expr: sp.Float) -> str:
        return f"scalar_t({float(expr):.17g})"

    def _print_Pow(self, expr: sp.Expr) -> str:
        # Pin the reciprocal spelling to the scalar type. C99CodePrinter has
        # rendered this as "1.0/x" in older SymPy and as "scalar_t(1)/x" in newer
        # releases (which routes the 1 through _print_Integer), so without this
        # override the generated text depends on the SymPy version.
        #
        # Note this is about output stability, NOT precision: "1.0/x" with a float
        # x is not in fact a double division in the emitted code. float->double->
        # float is exactly narrowable for +-*/, so the compiler contracts it; both
        # spellings were measured to produce byte-identical object code.
        if expr.exp == -1:
            from sympy.printing.precedence import PRECEDENCE
            return f"scalar_t(1)/{self.parenthesize(expr.base, PRECEDENCE['Mul'])}"
        return super()._print_Pow(expr)


def dof(node: int, field: int) -> int:
    return node * N_FIELD + field


def build_symbols() -> dict[str, object]:
    return {
        "rho": sp.Symbol("rho"),
        "mu": sp.Symbol("mu"),
        "det": sp.Symbol("det"),
        "cof": sp.symbols("cof0:9"),
        "ux": sp.symbols("ux0:8"),
        "uy": sp.symbols("uy0:8"),
        "uz": sp.symbols("uz0:8"),
        "p": sp.symbols("p0:8"),
        "sgn": sp.symbols("sgn0:12"),
    }


def area(sym: dict[str, object], ar: tuple[sp.Rational, sp.Rational, sp.Rational]) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    cof = sym["cof"]
    ar0, ar1, ar2 = ar
    return (
        cof[0] * ar0 + cof[3] * ar1 + cof[6] * ar2,
        cof[1] * ar0 + cof[4] * ar1 + cof[7] * ar2,
        cof[2] * ar0 + cof[5] * ar1 + cof[8] * ar2,
    )


def velocity_gradient(sym: dict[str, object]) -> tuple[sp.Expr, ...]:
    cof = sym["cof"]
    det = sym["det"]
    inv_det = 1 / det
    out: list[sp.Expr] = []
    for comp in (sym["ux"], sym["uy"], sym["uz"]):
        dr = sp.Integer(0)
        ds = sp.Integer(0)
        dt = sp.Integer(0)
        for a in range(N_NODE):
            dr += comp[a] * DN_REF[a][0]
            ds += comp[a] * DN_REF[a][1]
            dt += comp[a] * DN_REF[a][2]
        out.extend(
            (
                (cof[0] * dr + cof[3] * ds + cof[6] * dt) * inv_det,
                (cof[1] * dr + cof[4] * ds + cof[7] * dt) * inv_det,
                (cof[2] * dr + cof[5] * ds + cof[8] * dt) * inv_det,
            )
        )
    return tuple(out)


def face_residual_expr(sym: dict[str, object], s: int) -> tuple[list[sp.Expr], sp.Expr]:
    rho = sym["rho"]
    mu = sym["mu"]
    ux = sym["ux"]
    uy = sym["uy"]
    uz = sym["uz"]
    p = sym["p"]
    sgn = sym["sgn"]

    r = [sp.Integer(0)] * N_DOF
    g00, g01, g02, g10, g11, g12, g20, g21, g22 = velocity_gradient(sym)
    i, j, ar = SCS[s]
    ax, ay, az = area(sym, ar)
    adv_x = sp.Rational(1, 2) * (ux[i] + ux[j])
    adv_y = sp.Rational(1, 2) * (uy[i] + uy[j])
    adv_z = sp.Rational(1, 2) * (uz[i] + uz[j])
    mdot = rho * (adv_x * ax + adv_y * ay + adv_z * az)
    mdot_abs = sgn[s] * mdot
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
    return r, mdot


def residual_exprs(sym: dict[str, object]) -> tuple[list[sp.Expr], list[sp.Expr]]:
    r = [sp.Integer(0)] * N_DOF
    mdots: list[sp.Expr] = []
    for s in range(len(SCS)):
        fr, mdot = face_residual_expr(sym, s)
        mdots.append(mdot)
        for i in range(N_DOF):
            r[i] += fr[i]
    return r, mdots


def cse_code(exprs: list[sp.Expr], outputs: list[str], indent: str = "    ", op: str = "=") -> str:
    printer = ScalarPrinter()
    nonzero = [(expr, out) for expr, out in zip(exprs, outputs) if expr != 0]
    replacements, reduced = sp.cse([expr for expr, _out in nonzero], symbols=sp.numbered_symbols("x"), optimizations="basic")
    lines: list[str] = []
    for var, expr in replacements:
        lines.append(f"{indent}const scalar_t {var} = {printer.doprint(expr)};")
    for (_expr, out), expr in zip(nonzero, reduced):
        lines.append(f"{indent}{out} {op} {printer.doprint(expr)};")
    return "\n".join(lines)


def cse_atomic_add_code(exprs: list[sp.Expr], outputs: list[str], indent: str = "    ") -> str:
    printer = ScalarPrinter()
    nonzero = [(expr, out) for expr, out in zip(exprs, outputs) if expr != 0]
    replacements, reduced = sp.cse([expr for expr, _out in nonzero], symbols=sp.numbered_symbols("x"), optimizations="basic")
    lines: list[str] = []
    for var, expr in replacements:
        lines.append(f"{indent}const scalar_t {var} = {printer.doprint(expr)};")
    for k, ((_expr, out), expr) in enumerate(zip(nonzero, reduced)):
        lines.append(f"{indent}const scalar_t add{k} = {printer.doprint(expr)};")
        # CVFEM_ATOMIC_ADD expands to `#pragma omp atomic update` on a threaded host,
        # atomicAdd on the device, and a plain += when serial. Keeping the choice in
        # the macro is what lets these kernels compile for both targets unchanged.
        lines.append(f"{indent}CVFEM_ATOMIC_ADD({out}, add{k});")
    return "\n".join(lines)


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


def sign_locals(mdots: list[sp.Expr]) -> str:
    printer = ScalarPrinter()
    lines: list[str] = []
    for s, mdot in enumerate(mdots):
        lines.append(f"    const scalar_t mdot{s} = {printer.doprint(mdot)};")
        lines.append(
            f"    const scalar_t sgn{s} = mdot{s} > scalar_t(0) ? scalar_t(1) : "
            f"(mdot{s} < scalar_t(0) ? scalar_t(-1) : scalar_t(0));"
        )
    return "\n".join(lines)


def residual_outputs() -> list[str]:
    return [f"r[{i}]" for i in range(N_DOF)]


def jac_block_exprs(jac: list[sp.Expr], row_node: int, col_node: int) -> list[sp.Expr]:
    exprs: list[sp.Expr] = []
    for row_field in range(N_FIELD):
        row = dof(row_node, row_field)
        for col_field in range(N_FIELD):
            col = dof(col_node, col_field)
            exprs.append(jac[row * N_DOF + col])
    return exprs


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


def main() -> None:
    OUT.write_text(generate())


if __name__ == "__main__":
    main()
