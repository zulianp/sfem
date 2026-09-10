"""Shared machinery for the CVFEM SymPy kernel generators.

The HEX8 and TET4 synthesizers were a fork pair: the same C99 printer, the same CSE
emitters written out four times over, and -- most of all -- the same thirty-five lines of
CVFEM flux algebra, character for character in both files. That is not merely
duplication, it is a place where a physics fix can land in one element and silently not
the other. The version drift it caused is already on the record: the HEX8 printer grew a
``_print_Pow`` override that pins the reciprocal spelling to the scalar type, and the
TET4 printer never got it.

What lives here is what is genuinely element-independent: the printer, one parameterized
CSE emitter that subsumes the seven that existed, the DOF numbering, the sign locals, the
BSR block selection, and the sub-control-surface flux itself. What deliberately does not
live here is anything that knows about a particular element -- the SCS and shape-function
tables, HEX8's isoparametric per-face geometry and its quarantine partitioner, TET4's
SIMD lane machinery. Those stay with their element.

The contract for changing anything in this file is that the emitted headers must come out
byte for byte as they were, apart from the recorded SymPy version. The generators support
``--check`` for exactly that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy as sp
from sympy.printing.c import C99CodePrinter

# Velocity (3) plus pressure. Both elements are equal-order and mix all four fields into
# one 4x4 block, which is what makes the BSR slot addressing below element-independent.
N_FIELD = 4


# Whether ScalarPrinter pins the reciprocal spelling; see set_stable_pow(). Module state
# rather than a constructor argument because the printer is built deep inside the emitters
# and there is exactly one element being generated per process.
_STABLE_POW = True


def set_stable_pow(value: bool) -> None:
    """Select the reciprocal spelling, and therefore which bytes come out.

    With this on, ``1/x`` prints as ``scalar_t(1)/x``; with it off, C99CodePrinter's own
    rendering is used, which is ``1.0/x`` on SymPy 1.12 and ``scalar_t(1)/x`` on newer
    releases -- so *off* means the emitted text depends on the SymPy version.

    HEX8 has been generated with it on since the override was added there. TET4 has not,
    and turning it on changes 35 lines of ``cvfem_tet4_ns_upwind_sympy_kernels.hpp``,
    including several inside the SIMD paths where the reciprocal is taken against a
    vector type and the literal's type therefore selects an overload. The flipped header
    was checked and compiles, SIMD paths included; what has *not* been checked is whether
    it changes the emitted object code or the benchmark. That makes it a change to a
    measured kernel, which belongs in a commit with a measurement behind it rather than
    smuggled in with a refactor. See python/README.md.
    """
    global _STABLE_POW
    _STABLE_POW = value


class ScalarPrinter(C99CodePrinter):
    """C99 printer that spells every literal through the kernel's ``scalar_t``.

    The kernels are templates instantiated at both float and double, so a bare ``0.5``
    would silently promote the arithmetic around it.
    """

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
        if _STABLE_POW and expr.exp == -1:
            from sympy.printing.precedence import PRECEDENCE
            return f"scalar_t(1)/{self.parenthesize(expr.base, PRECEDENCE['Mul'])}"
        return super()._print_Pow(expr)


def dof(node: int, field: int, n_field: int = N_FIELD) -> int:
    return node * n_field + field


# --------------------------------------------------------------------------- emission

def cse_emit(exprs: list[sp.Expr],
             outputs: list[str],
             indent: str = "    ",
             *,
             mode: str = "assign",
             op: str = "=",
             drop_zeros: bool = False,
             prefix: str = "x") -> str:
    """Common-subexpression-eliminate ``exprs`` and emit them into ``outputs``.

    ``drop_zeros`` is not cosmetic and is not a default: removing the zero expressions
    changes what ``sp.cse`` is given, and therefore changes the temporaries it finds and
    the text it produces. Each call site keeps whichever behaviour it had.

    ``mode`` selects how a result is delivered:

    ``assign``        ``out <op> expr;``            -- ``op`` is ``=`` or ``+=``
    ``declare``       ``const scalar_t out = expr;``
    ``vector_store``  ``cvfem_store_scalar_v(out, expr);``, temporaries typed ``auto``
    ``atomic_add``    a named temporary, then ``CVFEM_ATOMIC_ADD(out, tmp);``
    """
    printer = ScalarPrinter()
    if drop_zeros:
        pairs = [(e, o) for e, o in zip(exprs, outputs) if e != 0]
    else:
        pairs = list(zip(exprs, outputs))

    # ``prefix`` exists so a kernel can run more than one CSE pass without the two sets of
    # temporaries colliding -- the two-level geometry arrangement factors the
    # geometry-only subexpressions in one pass and the field-dependent remainder in
    # another, and both would otherwise be called x0, x1, ...
    replacements, reduced = sp.cse([e for e, _o in pairs],
                                   symbols=sp.numbered_symbols(prefix),
                                   optimizations="basic")

    temp_type = "auto" if mode == "vector_store" else "scalar_t"
    lines: list[str] = []
    for var, expr in replacements:
        lines.append(f"{indent}const {temp_type} {var} = {printer.doprint(expr)};")

    for k, ((_expr, out), expr) in enumerate(zip(pairs, reduced)):
        text = printer.doprint(expr)
        if mode == "assign":
            lines.append(f"{indent}{out} {op} {text};")
        elif mode == "declare":
            lines.append(f"{indent}const scalar_t {out} = {text};")
        elif mode == "vector_store":
            lines.append(f"{indent}cvfem_store_scalar_v({out}, {text});")
        elif mode == "atomic_add":
            lines.append(f"{indent}const scalar_t add{k} = {text};")
            # CVFEM_ATOMIC_ADD expands to `#pragma omp atomic update` on a threaded host,
            # atomicAdd on the device, and a plain += when serial. Keeping the choice in
            # the macro is what lets these kernels compile for both targets unchanged.
            lines.append(f"{indent}CVFEM_ATOMIC_ADD({out}, add{k});")
        else:
            raise ValueError(f"unknown cse_emit mode: {mode}")
    return "\n".join(lines)


def sign_locals(mdots: list[sp.Expr], indent: str = "    ") -> str:
    """The mass flux per sub-control surface and its branch-free sign.

    The sign is a separate local rather than an ``Abs`` inside the flux because the
    Jacobian differentiates the flux, and a semismooth ``sgn`` frozen at the current
    iterate is what keeps that derivative defined at a flow reversal.
    """
    printer = ScalarPrinter()
    lines: list[str] = []
    for s, mdot in enumerate(mdots):
        lines.append(f"{indent}const scalar_t mdot{s} = {printer.doprint(mdot)};")
        lines.append(
            f"{indent}const scalar_t sgn{s} = mdot{s} > scalar_t(0) ? scalar_t(1) : "
            f"(mdot{s} < scalar_t(0) ? scalar_t(-1) : scalar_t(0));"
        )
    return "\n".join(lines)


def jac_block_exprs(jac: list[sp.Expr],
                    row_node: int,
                    col_node: int,
                    n_dof: int,
                    n_field: int = N_FIELD) -> list[sp.Expr]:
    """The n_field x n_field block of ``jac`` coupling ``col_node`` into ``row_node``,
    in row-major order -- the layout a BSR value slot expects."""
    exprs: list[sp.Expr] = []
    for row_field in range(n_field):
        row = dof(row_node, row_field, n_field)
        for col_field in range(n_field):
            col = dof(col_node, col_field, n_field)
            exprs.append(jac[row * n_dof + col])
    return exprs


# ---------------------------------------------------------------------- the flux itself

def face_flux_residual(*,
                       n_dof: int,
                       node_i: int,
                       node_j: int,
                       area: tuple[sp.Expr, sp.Expr, sp.Expr],
                       grad: tuple[sp.Expr, ...],
                       rho: sp.Expr,
                       mu: sp.Expr,
                       u: tuple[tuple[sp.Expr, ...], ...],
                       p: tuple[sp.Expr, ...],
                       sign: sp.Expr | None = None) -> tuple[list[sp.Expr], sp.Expr]:
    """The CVFEM flux across one sub-control surface, scattered to its two nodes.

    This is the physics, and it is identical for every element: what differs between
    HEX8 and TET4 is only how the outward area vector and the velocity gradient are
    obtained, both of which arrive here already built.

    ``area``  outward area vector of the sub-control surface, i -> j.
    ``grad``  the nine velocity-gradient components, row-major (g00 .. g22).
    ``u``     the three nodal velocity component tuples.
    ``sign``  the semismooth sign symbol for this surface; ``None`` uses ``Abs(mdot)``
              directly, which is the exact upwind switch but is not differentiable at a
              flow reversal.

    Returns the length-``n_dof`` residual contribution and the mass flux, the latter
    because the caller needs it to emit the sign locals.
    """
    ux, uy, uz = u
    g00, g01, g02, g10, g11, g12, g20, g21, g22 = grad
    ax, ay, az = area
    i, j = node_i, node_j

    r = [sp.Integer(0)] * n_dof

    adv_x = sp.Rational(1, 2) * (ux[i] + ux[j])
    adv_y = sp.Rational(1, 2) * (uy[i] + uy[j])
    adv_z = sp.Rational(1, 2) * (uz[i] + uz[j])
    mdot = rho * (adv_x * ax + adv_y * ay + adv_z * az)

    mdot_abs = sign * mdot if sign is not None else sp.Abs(mdot)
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


# ------------------------------------------------------------------------------- output

def add_output_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--out", type=Path, default=None,
                        help="write the header here instead of its default location")
    parser.add_argument("--check", action="store_true",
                        help="regenerate and diff against what is on disk, writing "
                             "nothing; exits non-zero if they differ")
    return parser


def emit(outputs: list[tuple[Path, str]], check: bool) -> int:
    """Write, or under ``--check`` compare, the generated headers.

    The comparison ignores the recorded SymPy version, which is the one line that is
    *supposed* to differ when the generator is run under a different SymPy than the
    committed headers were emitted with. Everything else differing means the CSE moved.
    """
    def strip_version(text: str) -> str:
        return "\n".join(l for l in text.split("\n") if not l.startswith("// SymPy "))

    if not check:
        for path, text in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"{path.name}: {text.count(chr(10))} lines")
        return 0

    rc = 0
    for path, text in outputs:
        if not path.exists():
            print(f"{path}: MISSING", file=sys.stderr)
            rc = 1
            continue
        if strip_version(path.read_text()) == strip_version(text):
            print(f"{path.name}: unchanged")
        else:
            print(f"{path.name}: DIFFERS from the checked-in header", file=sys.stderr)
            rc = 1
    return rc
