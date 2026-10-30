"""The inexact apply's generated half: what the library's stored-tangent operator calls.

The plans are `plans/point_tangent.py` (the material) and
`plans/inexact_element.py` (the element and its rule); this prints them.  Three
kinds of file, each with one owner:

    <unit>_<element>_inexact.hpp           the material at one point, bound to its element
    reference/inexact_<element>_q<n>.hpp   the element's stages and tables, any material
    reference/inexact_contract_d3.hpp      the contraction of one stored tangent, any element

The library (`src/frontend/ops/inexact/`) owns everything else -- gathers,
geometry, the encodings, the traversals -- and is the same for every material,
which is what lets one set of hand-measured containers serve them all.

The stages run in the precision the caller gives them.  The increment enters
relative to the element's first node on the tensor-product elements, so the
sweeps' rounding is relative to how much the increment varies over the element
rather than to its size; every sweep contains a derivative, which annihilates
the constant.
"""

from string import Template

import sympy as sp

from codegen.framework.emitters.cprinter import _sfem_ccode
from codegen.framework.emitters.quadrature_codegen import (
    cpp_scalar_literal,
    reference_header_path,
)
from codegen.framework.plans.form_transformations import stabilise_constant_cancellation
from codegen.framework.plans.inexact_element import (
    full_components,
    inexact_plans,
    packed_slot,
    symmetric_components,
)
from codegen.framework.plans.point_tangent import (
    flux_form_for_collection,
    geometry_symbols,
    point_tangent_parameters,
    physical_field_gradient_definitions,
    point_tangent,
    point_tangent_plan,
    point_tangent_quadrature_fields,
    reference_field_gradient_symbols,
)
from codegen.framework.plans.evaluation_strategy import (
    EvaluationStrategy,
    evaluation_strategy,
)
from codegen.framework.targets import current_target


# ----------------------------------------------------------------------------- printing


def _literal(value):
    """A double literal, exact for integers: `t(...)` is applied by the caller."""
    value = float(value)
    if value == int(value) and abs(value) < 1e6:
        return "%d" % int(value)
    return "%.17g" % value


def _row(values):
    return "{" + ", ".join(values) + "}"


def _double(value):
    """A double table entry, written as one: `1.0`, not `1`."""
    value = float(value)
    if value == int(value) and abs(value) < 1e6:
        return "%d.0" % int(value)
    return "%.17g" % value


#: C++ spellings of a boolean the plan decided.
_CPP_BOOL = {True: "true", False: "false"}


def _linear_combination(terms):
    """`sum c * s`, skipping zeros and folding +-1."""
    parts = []
    for coefficient, symbol in terms:
        coefficient = float(coefficient)
        if abs(coefficient) < 1e-15:
            continue
        if coefficient == 1.0:
            parts.append("+ %s" % symbol)
        elif coefficient == -1.0:
            parts.append("- %s" % symbol)
        elif coefficient < 0:
            parts.append("- t(%s)*%s" % (_literal(-coefficient), symbol))
        else:
            parts.append("+ t(%s)*%s" % (_literal(coefficient), symbol))
    if not parts:
        return "t(0)"
    text = " ".join(parts)
    return text[2:] if text.startswith("+ ") else "-" + text[2:]


def _qualifier():
    """`static SFEM_INLINE`, or the device target's spelling of it."""
    return "static %s" % current_target().inline_qualifier()


def _simd():
    """The bound target's vectorisation pragma, or nothing."""
    pragma = current_target().vectorize_pragma()
    return [pragma] if pragma else []


def _work_item_lines(lines):
    """The stages' work item, as the bound target spells it.

    The stages are written once, over the work item's placeholders: a lane of a
    block of VS elements on the CPU (`${E}` the `[VS]` extent, `${L}` the `[lane]`
    subscript, `${LOOP}` the vectorised lane loop, `${NE}`/`${NA}` the lane count
    parameter and argument, `${W}`/`${V}` the width template parameter and
    argument), the element itself on a device thread, where all of them are empty
    and the loop is a plain scope.
    """
    target = current_target()
    width = target.kernel_vector_width()
    spelled = dict(
        E=target.work_item_extent(),
        L=target.work_item_subscript(),
        W="".join(", int %s" % w for w in (width,) if w),
        V="".join(", %s" % w for w in (width,) if w),
        NE="".join("const int ne, " for _w in (width,) if _w),
        NA="".join("ne, " for _w in (width,) if _w),
    )
    out = []
    for line in lines:
        indent = line[: len(line) - len(line.lstrip())]
        loop = {"${LOOP}": lambda: list(target.work_item_loop_lines(indent))}
        out += loop.get(line.strip(), lambda: [Template(line).substitute(spelled)])()
    return out


def _assignment_lines(assignments, prefix, indent="  "):
    """Common subexpressions first, then the named values, as C declarations.

    One pass per stage: a later stage reads an earlier one's names, and a single
    pass over both would hoist a temporary above the definition it depends on.
    """
    if not assignments:
        return []
    symbols = [symbol for symbol, _expression in assignments]
    expressions = [
        stabilise_constant_cancellation(expression) for _symbol, expression in assignments
    ]
    temporaries, reduced = sp.cse(expressions, symbols=sp.numbered_symbols("%s_t" % prefix))
    lines = [
        "%sconst s_t %s = %s;" % (indent, symbol, _sfem_ccode(expression))
        for symbol, expression in temporaries
    ]
    lines.extend(
        "%sconst s_t %s = %s;" % (indent, symbol, _sfem_ccode(expression))
        for symbol, expression in zip(symbols, reduced)
    )
    return lines


# ------------------------------------------------------------------- the element


def element_header_path(plan):
    return reference_header_path("inexact_%s" % plan.rule_key)


def contract_header_path(dim):
    return reference_header_path("inexact_contract_d%d" % int(dim))


def _simplex_stage_lines(plan):
    """`stage_in` and `stage_out` from the derivative table, straight-line."""
    n, r_count, dim = plan.n_nodes, plan.n_gradients, plan.dim
    order = dim * dim
    gradients = plan.gradients
    q = _qualifier()
    lines = [
        "",
        "  // P[r][k*%d+m] = sum_j dphi_j/dxi_m (at r) h[k][j]: in the vector's precision, then" % dim,
        "  // rounded to the arithmetic type, so differences of nodal values are formed first.",
        "  template <typename h_t, typename t_out${W}>",
        "  %s void stage_in(${NE}const h_t h[%d][%d]${E}, t_out P[%d][%d]${E}) {"
        % (q, dim, n, r_count, order),
        "    using t = h_t;",
    ]
    lines.append("    ${LOOP}")
    for k in range(dim):
        for j in range(n):
            lines.append("      const t h%d_%d = h[%d][%d]${L};" % (k, j, k, j))
    for r in range(r_count):
        for k in range(dim):
            for m in range(dim):
                expression = _linear_combination(
                    [(gradients[r][m][j], "h%d_%d" % (k, j)) for j in range(n)]
                )
                lines.append("      P[%d][%d]${L} = t_out(%s);" % (r, k * dim + m, expression))
    lines += ["    }", "  }", ""]
    lines += [
        "  // out[i][p] = sum_r sum_n dphi_p/dxi_n (at r) Y[r][i*%d+n], accumulated in out_t." % dim,
        "  template <typename y_t, typename out_t${W}>",
        "  %s void stage_out(${NE}const y_t Y[%d][%d]${E}, out_t out[%d][%d]${E}) {"
        % (q, r_count, order, dim, n),
        "    using t = out_t;",
    ]
    lines.append("    ${LOOP}")
    for r in range(r_count):
        for c in range(order):
            lines.append("      const t y%d_%d = t(Y[%d][%d]${L});" % (r, c, r, c))
    for i in range(dim):
        for p in range(n):
            expression = _linear_combination(
                [
                    (gradients[r][nn][p], "y%d_%d" % (r, i * dim + nn))
                    for r in range(r_count)
                    for nn in range(dim)
                ]
            )
            lines.append("      out[%d][%d]${L} = %s;" % (i, p, expression))
    lines += ["    }", "  }"]
    return lines


def _linear_hex_stage_lines(plan):
    """HEX8 on 2x2x2 points by one-dimensional sweeps, eliminated by hand.

    With two linear shape functions the derivative along an axis is the
    difference of the two nodal values, constant along that axis, and the value
    at a point is `u0 + x_q (u1 - u0)`; both facts are written out so the
    elimination is in the code rather than left to the compiler.
    """
    line = plan.line
    assert line.n_points == 2 and line.n_shape == 2
    for q in range(2):
        x = line.points[q]
        assert abs(line.values[q][0] - (1 - x)) < 1e-14 and abs(line.values[q][1] - x) < 1e-14
        assert line.derivatives[q] == (-1.0, 1.0)
    g0, g1 = "%.17g" % line.points[0], "%.17g" % line.points[1]
    q = _qualifier()
    lines = [
        "",
        "  template <typename h_t, typename t_out${W}>",
        "  %s void stage_in(${NE}const h_t h[3][8]${E}, t_out P[8][9]${E}) {" % q,
        "    using t = h_t;",
        "    const t g0 = t(%s), g1 = t(%s);" % (g0, g1),
    ]
    lines.append("    ${LOOP}")
    for k in range(3):
        def u(sx, sy, sz, k=k):
            return "h[%d][%d]${L}" % (k, sx + 2 * sy + 4 * sz)

        p = "c%d_" % k
        for sy in (0, 1):
            for sz in (0, 1):
                lines.append("      const t %sdx%d%d = %s - %s;" % (p, sy, sz, u(1, sy, sz), u(0, sy, sz)))
                for qx, gq in ((0, "g0"), (1, "g1")):
                    lines.append(
                        "      const t %svx%d%d%d = %s + %s*%sdx%d%d;"
                        % (p, qx, sy, sz, u(0, sy, sz), gq, p, sy, sz)
                    )
        for sz in (0, 1):
            lines.append("      const t %sddx%d = %sdx1%d - %sdx0%d;" % (p, sz, p, sz, p, sz))
            for qy, gq in ((0, "g0"), (1, "g1")):
                lines.append(
                    "      const t %sdxy%d%d = %sdx0%d + %s*%sddx%d;" % (p, qy, sz, p, sz, gq, p, sz)
                )
            for qx in (0, 1):
                lines.append(
                    "      const t %sdy%d%d = %svx%d1%d - %svx%d0%d;" % (p, qx, sz, p, qx, sz, p, qx, sz)
                )
                for qy, gq in ((0, "g0"), (1, "g1")):
                    lines.append(
                        "      const t %svxy%d%d%d = %svx%d0%d + %s*%sdy%d%d;"
                        % (p, qx, qy, sz, p, qx, sz, gq, p, qx, sz)
                    )
        for qz, gq in ((0, "g0"), (1, "g1")):
            for qy in (0, 1):
                for qx in (0, 1):
                    point = qx + 2 * qy + 4 * qz
                    gx = "%sdxy%d0 + %s*(%sdxy%d1 - %sdxy%d0)" % (p, qy, gq, p, qy, p, qy)
                    gy = "%sdy%d0 + %s*(%sdy%d1 - %sdy%d0)" % (p, qx, gq, p, qx, p, qx)
                    gz = "%svxy%d%d1 - %svxy%d%d0" % (p, qx, qy, p, qx, qy)
                    lines.append("      P[%d][%d]${L} = t_out(%s);" % (point, k * 3 + 0, gx))
                    lines.append("      P[%d][%d]${L} = t_out(%s);" % (point, k * 3 + 1, gy))
                    lines.append("      P[%d][%d]${L} = t_out(%s);" % (point, k * 3 + 2, gz))
    lines += ["    }", "  }", ""]
    lines += [
        "  template <typename y_t, typename out_t${W}>",
        "  %s void stage_out(${NE}const y_t Y[8][9]${E}, out_t out[3][8]${E}) {" % q,
        "    using t = out_t;",
        "    const t g0 = t(%s), g1 = t(%s);" % (g0, g1),
    ]
    lines.append("    ${LOOP}")

    def transposed(a0, a1, name):
        # The transpose of the interpolation: n1 = g0 a0 + g1 a1, n0 = (a0 + a1) - n1.
        return [
            "      const t %s1 = g0*%s + g1*%s;" % (name, a0, a1),
            "      const t %s0 = (%s + %s) - %s1;" % (name, a0, a1, name),
        ]

    for i in range(3):
        p = "o%d_" % i

        def yv(qx, qy, qz, nn, i=i):
            return "t(Y[%d][%d]${L})" % (qx + 2 * qy + 4 * qz, i * 3 + nn)

        accumulated = {s: [] for s in range(8)}
        for qy in (0, 1):
            for qz in (0, 1):
                lines.append("      const t %sax%d%d = %s + %s;" % (p, qy, qz, yv(0, qy, qz, 0), yv(1, qy, qz, 0)))
        for qy in (0, 1):
            lines += transposed("%sax%d0" % (p, qy), "%sax%d1" % (p, qy), "%sbx%d_" % (p, qy))
        for sz in (0, 1):
            lines += transposed("%sbx0_%d" % (p, sz), "%sbx1_%d" % (p, sz), "%scx%d_" % (p, sz))
        for sy in (0, 1):
            for sz in (0, 1):
                c = "%scx%d_%d" % (p, sz, sy)
                accumulated[0 + 2 * sy + 4 * sz].append("- %s" % c)
                accumulated[1 + 2 * sy + 4 * sz].append("+ %s" % c)
        for qx in (0, 1):
            for qz in (0, 1):
                lines.append("      const t %say%d%d = %s + %s;" % (p, qx, qz, yv(qx, 0, qz, 1), yv(qx, 1, qz, 1)))
        for qx in (0, 1):
            lines += transposed("%say%d0" % (p, qx), "%say%d1" % (p, qx), "%sby%d_" % (p, qx))
        for sz in (0, 1):
            lines += transposed("%sby0_%d" % (p, sz), "%sby1_%d" % (p, sz), "%scy%d_" % (p, sz))
        for sx in (0, 1):
            for sz in (0, 1):
                c = "%scy%d_%d" % (p, sz, sx)
                accumulated[sx + 0 + 4 * sz].append("- %s" % c)
                accumulated[sx + 2 + 4 * sz].append("+ %s" % c)
        for qx in (0, 1):
            for qy in (0, 1):
                lines.append("      const t %saz%d%d = %s + %s;" % (p, qx, qy, yv(qx, qy, 0, 2), yv(qx, qy, 1, 2)))
        for qx in (0, 1):
            lines += transposed("%saz%d0" % (p, qx), "%saz%d1" % (p, qx), "%sbz%d_" % (p, qx))
        for sy in (0, 1):
            lines += transposed("%sbz0_%d" % (p, sy), "%sbz1_%d" % (p, sy), "%scz%d_" % (p, sy))
        for sx in (0, 1):
            for sy in (0, 1):
                c = "%scz%d_%d" % (p, sy, sx)
                accumulated[sx + 2 * sy + 0].append("- %s" % c)
                accumulated[sx + 2 * sy + 4].append("+ %s" % c)
        for s in range(8):
            expression = " ".join(accumulated[s])
            expression = expression[2:] if expression.startswith("+ ") else "-" + expression[2:]
            lines.append("      out[%d][%d]${L} = %s;" % (i, s, expression))
    lines += ["    }", "  }"]
    return lines


def _tensor_stage_lines(plan):
    """Sum-factorised stages for a tensor-product element of any order.

    One lane loop per one-dimensional sweep, the intermediates in small arrays
    rather than live temporaries, so no sweep holds more than a few values per
    lane.  The increment enters relative to the element's first node.
    """
    line = plan.line
    n1, q1 = line.n_shape, line.n_points
    values, derivatives = line.values, line.derivatives
    n, n_points = n1**3, q1**3
    q = _qualifier()

    def node(sx, sy, sz):
        return sx + n1 * (sy + n1 * sz)

    def lit(c):
        return "t(%s)" % _literal(c)

    lines = [
        "",
        "  template <typename h_t, typename t_out${W}>",
        "  %s void stage_in(${NE}const h_t h[3][%d]${E}, t_out P[%d][9]${E}) {" % (q, n, n_points),
        "    using t = t_out;",
        "    t u[%d]${E}, v[%d][%d][%d]${E}, d[%d][%d][%d]${E};" % (n, q1, n1, n1, q1, n1, n1),
        "    t vv[%d][%d][%d]${E}, dv[%d][%d][%d]${E}, vd[%d][%d][%d]${E};"
        % (q1, q1, n1, q1, q1, n1, q1, q1, n1),
        "    for (int k = 0; k < 3; ++k) {",
    ]
    lines.append("      ${LOOP}")
    for s in range(n):
        lines.append("        u[%d]${L} = t(h[k][%d]${L} - h[k][0]${L});" % (s, s))
    lines.append("      }")
    lines.append("      ${LOOP}")
    for sz in range(n1):
        for sy in range(n1):
            for qx in range(q1):
                us = ["u[%d]${L}" % node(sx, sy, sz) for sx in range(n1)]
                lines.append(
                    "        v[%d][%d][%d]${L} = %s;"
                    % (qx, sy, sz, " + ".join("%s*%s" % (lit(values[qx][sx]), us[sx]) for sx in range(n1)))
                )
                lines.append(
                    "        d[%d][%d][%d]${L} = %s;"
                    % (qx, sy, sz, " + ".join("%s*%s" % (lit(derivatives[qx][sx]), us[sx]) for sx in range(n1)))
                )
    lines.append("      }")
    lines.append("      ${LOOP}")
    for sz in range(n1):
        for qx in range(q1):
            for qy in range(q1):
                for name, table, source in (("vv", values, "v"), ("dv", values, "d"), ("vd", derivatives, "v")):
                    lines.append(
                        "        %s[%d][%d][%d]${L} = %s;"
                        % (
                            name, qx, qy, sz,
                            " + ".join(
                                "%s*%s[%d][%d][%d]${L}" % (lit(table[qy][sy]), source, qx, sy, sz)
                                for sy in range(n1)
                            ),
                        )
                    )
    lines.append("      }")
    lines.append("      ${LOOP}")
    for qz in range(q1):
        for qy in range(q1):
            for qx in range(q1):
                point = qx + q1 * (qy + q1 * qz)
                gx = " + ".join("%s*dv[%d][%d][%d]${L}" % (lit(values[qz][sz]), qx, qy, sz) for sz in range(n1))
                gy = " + ".join("%s*vd[%d][%d][%d]${L}" % (lit(values[qz][sz]), qx, qy, sz) for sz in range(n1))
                gz = " + ".join("%s*vv[%d][%d][%d]${L}" % (lit(derivatives[qz][sz]), qx, qy, sz) for sz in range(n1))
                lines.append("        P[%d][k * 3 + 0]${L} = %s;" % (point, gx))
                lines.append("        P[%d][k * 3 + 1]${L} = %s;" % (point, gy))
                lines.append("        P[%d][k * 3 + 2]${L} = %s;" % (point, gz))
    lines += ["      }", "    }", "  }", ""]

    def yq(qx, qy, qz, nn):
        return "Y[%d][i * 3 + %d]${L}" % (qx + q1 * (qy + q1 * qz), nn)

    lines += [
        "  template <typename y_t, typename out_t${W}>",
        "  %s void stage_out(${NE}const y_t Y[%d][9]${E}, out_t out[3][%d]${E}) {" % (q, n_points, n),
        "    using t = y_t;",
        "    t ax[%d][%d][%d]${E}, ay[%d][%d][%d]${E}, az[%d][%d][%d]${E};"
        % (q1, q1, n1, q1, q1, n1, q1, q1, n1),
        "    t bx[%d][%d][%d]${E}, c[%d][%d][%d]${E};" % (q1, n1, n1, q1, n1, n1),
        "    for (int i = 0; i < 3; ++i) {",
    ]
    lines.append("      ${LOOP}")
    for qx in range(q1):
        for qy in range(q1):
            for sz in range(n1):
                lines.append(
                    "        ax[%d][%d][%d]${L} = %s;"
                    % (qx, qy, sz, " + ".join("%s*%s" % (lit(values[qz][sz]), yq(qx, qy, qz, 0)) for qz in range(q1)))
                )
                lines.append(
                    "        ay[%d][%d][%d]${L} = %s;"
                    % (qx, qy, sz, " + ".join("%s*%s" % (lit(values[qz][sz]), yq(qx, qy, qz, 1)) for qz in range(q1)))
                )
                lines.append(
                    "        az[%d][%d][%d]${L} = %s;"
                    % (qx, qy, sz, " + ".join("%s*%s" % (lit(derivatives[qz][sz]), yq(qx, qy, qz, 2)) for qz in range(q1)))
                )
    lines.append("      }")
    lines.append("      ${LOOP}")
    for qx in range(q1):
        for sy in range(n1):
            for sz in range(n1):
                lines.append(
                    "        bx[%d][%d][%d]${L} = %s;"
                    % (qx, sy, sz, " + ".join("%s*ax[%d][%d][%d]${L}" % (lit(values[qy][sy]), qx, qy, sz) for qy in range(q1)))
                )
                lines.append(
                    "        c[%d][%d][%d]${L} = %s;"
                    % (
                        qx, sy, sz,
                        " + ".join(
                            "%s*ay[%d][%d][%d]${L} + %s*az[%d][%d][%d]${L}"
                            % (lit(derivatives[qy][sy]), qx, qy, sz, lit(values[qy][sy]), qx, qy, sz)
                            for qy in range(q1)
                        ),
                    )
                )
    lines.append("      }")
    lines.append("      ${LOOP}")
    for sz in range(n1):
        for sy in range(n1):
            for sx in range(n1):
                expression = " + ".join(
                    "%s*bx[%d][%d][%d]${L} + %s*c[%d][%d][%d]${L}"
                    % (lit(derivatives[qx][sx]), qx, sy, sz, lit(values[qx][sx]), qx, sy, sz)
                    for qx in range(q1)
                )
                lines.append("        out[i][%d]${L} = out_t(%s);" % (node(sx, sy, sz), expression))
    lines += ["      }", "    }", "  }"]
    return lines


#: The stages, by how the plan says they are written.
_STAGES_BY_FORM = {
    "tabulated": lambda plan: _simplex_stage_lines(plan),
    "linear_sweeps": lambda plan: _linear_hex_stage_lines(plan),
    "sweeps": lambda plan: _tensor_stage_lines(plan),
}


def _setup_stage_lines(plan):
    """`setup_in`: the reference gradient at the set-up's points, G[q][k*dim+m].

    The stored points are the set-up's own except for the moments, which
    integrate over the rule's points.
    """
    n, dim = plan.n_nodes, plan.dim
    q = _qualifier()
    lines = [
        "",
        "  // The reference gradient at the set-up's points: G[q][k*%d+m]." % dim,
        "  template <typename h_t, typename t_out${W}>",
        "  %s void setup_in(${NE}const h_t h[%d][%d]${E}, t_out G[NQ][%d]${E}) {"
        % (q, dim, n, dim * dim),
    ]
    return lines + _SETUP_BODY_BY_FORM[plan.setup_form](plan)


def _setup_from_the_stages(plan):
    return ["    stage_in<h_t, t_out${V}>(${NA}h, G);", "  }"]


def _setup_from_the_rule(plan):
    n, dim = plan.n_nodes, plan.dim
    lines = ["    using t = h_t;"]
    lines.append("    ${LOOP}")
    for k in range(dim):
        for j in range(n):
            lines.append("      const t h%d_%d = h[%d][%d]${L};" % (k, j, k, j))
    for point, gradient in enumerate(plan.setup_gradients):
        for k in range(dim):
            for m in range(dim):
                expression = _linear_combination([(gradient[m][j], "h%d_%d" % (k, j)) for j in range(n)])
                lines.append("      G[%d][%d]${L} = t_out(%s);" % (point, k * dim + m, expression))
    lines += ["    }", "  }"]
    return lines


_SETUP_BODY_BY_FORM = {"stages": _setup_from_the_stages, "explicit": _setup_from_the_rule}


def _fold_struct_lines(plan, symmetric):
    """The folded store's tables for a tangent of one packing.

    What is data: each slot's direction, the slots each mask keeps, where each
    stored tensor's numbers start, and the folded reference weights.  The masks,
    the sources and the weights per point follow from the lattice
    (`sfem_InexactElement.hpp`), so they are not tabulated.
    """
    layout = plan.fold_layout(symmetric)
    n_slots = layout.n_slots
    masks = 1 << plan.dim
    return [
        "  struct fold_%s {" % ("symmetric" if symmetric else "full"),
        "    static constexpr int NC = %d;" % n_slots,
        "    static constexpr int DIR[%d] = %s;" % (n_slots, _row(str(d) for d in layout.directions)),
        "    static constexpr int NKEEP[%d] = %s;" % (masks, _row(str(len(k)) for k in layout.keep)),
        "    static constexpr int KEEP[%d][%d] = {%s};"
        % (
            masks, n_slots,
            ", ".join(_row(str(v) for v in k + (0,) * (n_slots - len(k))) for k in layout.keep),
        ),
        "    static constexpr int OFF[%d] = %s;" % (plan.n_stored + 1, _row(str(o) for o in layout.offsets)),
        "    static constexpr int NS = %d;  // stored numbers per element" % layout.n_numbers,
        "    // The reference weight of each stored number after the fold.",
        "    static constexpr double XC[%d][%d] = {%s};"
        % (plan.n_stored, n_slots, ", ".join(_row(_double(v) for v in r) for r in layout.folded_weight)),
        "  };",
    ]


def _fold_lines(plan):
    """The fold on a tensor-product element: its lattice and its tables, per packing."""
    return [
        "",
        "  // The fold: along each line of points, the point with index DROP keeps no m = n = d",
        "  // components; they move to the line's other points, weighted by COEF1 of their index.",
        "  static constexpr int Q1 = %d;" % plan.line.n_points,
        "  static constexpr int DROP = %d;" % plan.fold.drop,
        "  static constexpr double COEF1[Q1] = %s;" % _row(_double(c) for c in plan.line_coefficients),
    ] + _fold_struct_lines(plan, True) + _fold_struct_lines(plan, False)


#: Only a tensor-product element folds.
_FOLD_BY_FOLDABILITY = {True: _fold_lines, False: lambda plan: []}


def _line_table_lines(plan):
    """The one-dimensional basis at the line's points, for the device's cooperative sweeps.

    A device thread per point cannot run the sum-factorised stages, which sweep a
    whole element; the element's threads run the sweeps together from these.
    """
    line = plan.line
    return [
        "",
        "  // The line's basis at its points, for the device's cooperative sweeps: B1[q][s] = phi_s(x_q),",
        "  // D1[q][s] = phi_s'(x_q).",
        "  static constexpr int S1 = %d;" % line.n_shape,
        "  static constexpr double B1[Q1][S1] = {%s};" % ", ".join(_row(_double(v) for v in row) for row in line.values),
        "  static constexpr double D1[Q1][S1] = {%s};"
        % ", ".join(_row(_double(v) for v in row) for row in line.derivatives),
    ]


#: Only the device reads the line's tables, and only where the host sweeps.
_LINE_TABLES = {("sweeps", True): _line_table_lines}


def _pair_lines(plan):
    """The moments' pairing: which gradient each stored tensor reads and writes, and how often
    it occurs in the symmetric sum."""
    return [
        "  static constexpr int PA[NV] = %s;" % _row(str(a) for a, _ in plan.pairs),
        "  static constexpr int PB[NV] = %s;" % _row(str(b) for _, b in plan.pairs),
        "  static constexpr double MULT[NV] = %s;" % _row(_double(x) for x in plan.multiplicity),
        "  static constexpr double FACTOR[NQ][NV] = {%s};"
        % ", ".join(_row(_double(v) for v in row) for row in plan.moment_factor),
    ]


#: A point store pairs tensor j with point j, which needs no table.
_PAIRS_BY_KIND = {"moments": _pair_lines, "points": lambda plan: []}


def _shape_order_lines(plan):
    return [
        "  // The mesh node each staged node reads.",
        "  static constexpr int SHAPE_ORDER[N] = %s;" % _row(str(s) for s in plan.shape_order),
    ]


_SHAPE_ORDER_BY_REORDERING = {True: _shape_order_lines, False: lambda plan: []}


def element_header_source(plan):
    """The element's struct: its constants, data tables, stages and fold."""
    n, dim = plan.n_nodes, plan.dim
    guard = "SFEM_CODEGEN_INEXACT_%s_%s" % (plan.rule_key.upper(), current_target().header_guard_suffix())
    gradients = ", ".join(
        _row(_row(_double(plan.gradients[r][m][j]) for j in range(n)) for m in range(dim))
        for r in range(plan.n_gradients)
    )
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
        "// The stored tangent's element half for %s on its %d-point rule (plans/inexact_element.py)."
        % (plan.element_type, plan.n_setup_points),
        "struct %s {" % plan.struct_name,
        '  static constexpr const char *name = "%s";' % plan.rule_key,
        "  static constexpr int DIM = %d;" % dim,
        "  static constexpr int N = %d;   // nodes" % n,
        "  static constexpr int R = %d;   // gradient vectors the store contracts with" % plan.n_gradients,
        "  static constexpr int NV = %d;  // stored tensors per element" % plan.n_stored,
        "  static constexpr int NQ = %d;  // quadrature points of the set-up" % plan.n_setup_points,
        "  static constexpr bool moments = %s;" % _CPP_BOOL[plan.is_moments],
        "  static constexpr bool foldable = %s;" % _CPP_BOOL[plan.foldable],
        "  static constexpr bool reordered = %s;" % _CPP_BOOL[plan.reordered],
        "  static constexpr double KVOL = %s;" % _double(plan.volume),
    ]
    lines += _SHAPE_ORDER_BY_REORDERING[plan.reordered](plan)
    lines += [
        "  // DT[r][m][j] = dphi_j/dxi_m at gradient point r.",
        "  static constexpr double DT[R][%d][N] = {%s};" % (dim, gradients),
        "  // Each stored tensor's reference weight (what the element mean contributes to it).",
        "  static constexpr double X[NV] = %s;" % _row(_double(x) for x in plan.reference_weight),
        "  static constexpr double QW[NQ] = %s;" % _row(_double(w) for w in plan.setup_weights),
    ]
    lines += _PAIRS_BY_KIND[plan.kind](plan)
    lines += _work_item_lines(_STAGES_BY_FORM[plan.stage_form](plan) + _setup_stage_lines(plan))
    lines += _FOLD_BY_FOLDABILITY[plan.foldable](plan)
    lines += _LINE_TABLES.get(
        (plan.stage_form, current_target().kernel_vector_width() is None), lambda plan: []
    )(plan)
    lines += ["};", "", "}  // namespace codegen", "}  // namespace sfem", "", "#endif", ""]
    return "\n".join(lines)


# --------------------------------------------------------------- the contraction


def _symmetric_slot_lines(order, slot, qualifier):
    """Row r, column c of the upper-triangle packing: data, so a table."""
    return [
        "  // Row r, column c of the packed tangent.",
        "  static constexpr int SLOT[%d][%d] = {%s};"
        % (order, order, ", ".join(_row(str(slot(r, c)) for c in range(order)) for r in range(order))),
        "  static constexpr int slot(const int r, const int c) { return SLOT[r][c]; }",
    ]


def _full_slot_lines(order, slot, qualifier):
    """Row r, column c of the full packing: row-major."""
    return [
        "  // Row r, column c of the packed tangent.",
        "  static constexpr int slot(const int r, const int c) { return r * %d + c; }" % order,
    ]


_SLOT_LINES_BY_SYMMETRY = {True: _symmetric_slot_lines, False: _full_slot_lines}


def _lane_contraction_lines(dim, n_slots, slot, q):
    """The contraction for a block of lanes, lane loop innermost, and its folded forms."""
    order = dim * dim
    lines = [
        "",
        "  // The same for a block of lanes, one loop over them.",
        "  template <typename t${W}>",
        "  %s void contract_lanes(${NE}const t S[%d]${E}, const t G[%d]${E}, t Y[%d]${E}) {"
        % (q, n_slots, order, order),
    ]
    lines.append("    ${LOOP}")
    for r in range(order):
        lines.append(
            "      Y[%d]${L} += %s;"
            % (r, " + ".join("S[%d]${L}*G[%d]${L}" % (slot(r, c), c) for c in range(order)))
        )
    lines.append("    }")
    lines += [
        "  }",
        "",
        "  // The folded store's contraction: bit d of MASK drops the m = n = d terms.",
        "  template <int MASK, typename t${W}>",
        "  %s void contract_lanes_masked(${NE}const t S[%d]${E}, const t G[%d]${E}, t Y[%d]${E}) {"
        % (q, n_slots, order, order),
    ]
    for mask in range(1 << dim):
        lines.append("    %s constexpr (MASK == %d) {" % ("if" if mask == 0 else "} else if", mask))
        lines.append("      ${LOOP}")
        for r in range(order):
            terms = " + ".join(
                "S[%d]${L}*G[%d]${L}" % (slot(r, c), c)
                for c in range(order)
                if not (r % dim == c % dim and (mask >> (r % dim)) & 1)
            )
            lines.append("        Y[%d]${L} += %s;" % (r, terms))
        lines.append("      }")
    lines += [
        "    }",
        "  }",
        "",
        "  template <typename t${W}>",
        "  %s void contract_lanes_masked(const int mask, ${NE}const t S[%d]${E}, const t G[%d]${E}, t Y[%d]${E}) {"
        % (q, n_slots, order, order),
        "    switch (mask) {",
    ]
    for mask in range(1 << dim):
        lines.append("      case %d: contract_lanes_masked<%d, t${V}>(${NA}S, G, Y); break;" % (mask, mask))
    return _work_item_lines(lines + ["    }", "  }"])


def _scalar_masked_contraction_lines(dim, n_slots, slot, q):
    """A device thread's folded contraction: bit d of MASK drops the m = n = d terms.

    A thread holds one element, so there are no lanes; the unmasked contraction
    is `contract` itself.
    """
    order = dim * dim
    lines = [
        "",
        "  // The folded store's contraction: bit d of MASK drops the m = n = d terms.",
        "  template <int MASK, typename t>",
        "  %s void contract_masked(const t *const S, const t *const g, t *const y) {" % q,
    ]
    for mask in range(1 << dim):
        lines.append("    %s constexpr (MASK == %d) {" % ("if" if mask == 0 else "} else if", mask))
        for r in range(order):
            terms = " + ".join(
                "S[%d]*g[%d]" % (slot(r, c), c)
                for c in range(order)
                if not (r % dim == c % dim and (mask >> (r % dim)) & 1)
            )
            lines.append("      y[%d] += %s;" % (r, terms))
    return lines + ["    }", "  }"]


#: A target whose work item is a lane of a block gets the lane forms; a device
#: thread holds one element and contracts it with `contract` and `contract_masked`.
_WORK_ITEM_CONTRACTIONS = {False: _lane_contraction_lines, True: _scalar_masked_contraction_lines}


def contract_header_source(dim):
    """`y[r] += sum_c S[slot(r, c)] g[c]` for one stored tangent, both packings.

    Straight-line: 81 multiply-adds per point, whatever the packing, with the
    symmetric packing reading each off-diagonal number twice.  The masked forms
    drop the terms a folded point does not store.
    """
    order = dim * dim
    q = _qualifier()
    guard = "SFEM_CODEGEN_INEXACT_CONTRACT_D%d_%s" % (dim, current_target().header_guard_suffix())
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <bool SYM>",
        "struct inexact_contract_d%d;" % dim,
    ]
    for symmetric in (True, False):
        n_slots = symmetric_components(dim) if symmetric else full_components(dim)

        def slot(r, c, symmetric=symmetric):
            return packed_slot(r, c, order, symmetric)

        lines += [
            "",
            "template <>",
            "struct inexact_contract_d%d<%s> {" % (dim, _CPP_BOOL[symmetric]),
            "  static constexpr int NC = %d;" % n_slots,
        ] + _SLOT_LINES_BY_SYMMETRY[symmetric](order, slot, q) + [
            "",
            "  template <typename t>",
            "  %s void contract(const t *const S, const t *const g, t *const y) {" % q,
        ]
        for r in range(order):
            lines.append(
                "    y[%d] += %s;" % (r, " + ".join("S[%d]*g[%d]" % (slot(r, c), c) for c in range(order)))
            )
        lines += ["  }"]
        lines += _WORK_ITEM_CONTRACTIONS[current_target().kernel_vector_width() is None](
            dim, n_slots, slot, q
        )
        lines += ["};"]
    lines += ["", "}  // namespace codegen", "}  // namespace sfem", "", "#endif", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------- the material


#: The prefix of the integrand's temporaries, per evaluation strategy.  It is not
#: cosmetic: sympy prints a sum's terms in the order of their symbols' names, so
#: the prefix decides the order the additions are rounded in.  These are the
#: prefixes the projected tangent kernels used, so the stored tangents reproduce
#: the measured ones bit for bit.
_INTEGRAND_PREFIX = {
    EvaluationStrategy.EXPANDED: "tangent",
    EvaluationStrategy.QUADRATURE: "integrand",
    EvaluationStrategy.SUM_FACTORIZED: "integrand",
}


def _point_lines(plan, units, parameters, element_type):
    """The point tangent's body: the sum of the units' tangents at one point."""
    dim = plan.dim
    adjugate, determinant = geometry_symbols(dim)
    total = [sp.Integer(0)] * plan.tangent_components
    gradient_symbols = previous_symbols = None
    for flux_form, is_deformation_gradient in units:
        tangent, gradient_symbols, previous_symbols = point_tangent(
            plan, flux_form, adjugate, determinant, is_deformation_gradient
        )
        total = [a + b for a, b in zip(total, tangent)]
    reads_state = bool(plan.state_dependence(total, gradient_symbols))
    reads_previous = bool(plan.state_dependence(total, previous_symbols))

    lines = []
    for index, name in enumerate(parameters):
        lines.append("  const s_t %s = parameters[%d];" % (name, index))
    # An integration-point field's components at this point, as the forms name them.
    for index, (name, _components) in enumerate(point_tangent_quadrature_fields(form for form, _ in units)):
        lines.append("  const s_t *const RSTR %s_qp = quadrature[%d];" % (name, index))
    stage = []
    for role, symbols, read in (
        ("gu", gradient_symbols, reads_state),
        ("gz", previous_symbols, reads_previous),
    ):
        if not read:
            continue
        reference = reference_field_gradient_symbols(dim, role)
        for c in range(dim):
            for a in range(dim):
                lines.append("  const s_t %s = %s_ref[%d];" % (reference[c][a], role, c * dim + a))
        stage.extend(
            physical_field_gradient_definitions(reference, symbols, adjugate, determinant, dim)
        )
    used = set()
    for expression in list(total) + [expression for _symbol, expression in stage]:
        used |= {str(s) for s in expression.free_symbols}
    for index in range(dim * dim):
        if "adjugate%d" % index in used:
            lines.append("  const s_t adjugate%d = adjugate[%d];" % (index, index))
    if "determinant" not in used:
        lines.append("  (void)determinant;")
    lines.extend(_assignment_lines(stage, "gradient"))
    prefix = _INTEGRAND_PREFIX[evaluation_strategy(element_type)]
    names = [sp.Symbol("%s%d" % (prefix, slot)) for slot in range(plan.tangent_components)]
    lines.extend(_assignment_lines(list(zip(names, total)), prefix))
    lines.extend(
        "  tangent[%d] = %s%d;" % (slot, prefix, slot) for slot in range(plan.tangent_components)
    )
    return lines, reads_state, reads_previous


#: Each integration-point field's component count, where the tangent reads any.
_QUADRATURE_COMPONENT_LINES = {
    True: lambda fields: [
        "  static constexpr int quadrature_components[%d] = %s;" % (len(fields), _row(str(c) for _n, c in fields))
    ],
    False: lambda fields: [],
}


def unit_header_source(unit_name, element_plan, units, parameters):
    """The material's tangent at one point, bound to its element.

    `units` are `(flux_form, is_deformation_gradient)` pairs whose tangents are
    summed; `parameters` are the material constants, in the order the operator
    passes them.
    """
    plan = point_tangent_plan(form for form, _ in units)
    fields = point_tangent_quadrature_fields(form for form, _ in units)
    body, reads_state, reads_previous = _point_lines(
        plan, units, parameters, element_plan.element_type
    )
    struct = "%s_%s_inexact" % (unit_name, element_plan.element_type.lower())
    guard = "SFEM_CODEGEN_%s_%s" % (struct.upper(), current_target().header_guard_suffix())
    q = _qualifier()
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        '#include "%s"' % current_target().header_name("kernel_math"),
        '#include "%s"' % element_header_path(element_plan).rsplit("/", 1)[-1],
        '#include "%s"' % contract_header_path(plan.dim).rsplit("/", 1)[-1],
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
        "// %s on %s: the weighted tangent the stored-tangent operator keeps at every point."
        % (unit_name, element_plan.element_type),
        "struct %s {" % struct,
        "  using element = %s;" % element_plan.struct_name,
        "  using contraction = inexact_contract_d%d<%s>;" % (plan.dim, _CPP_BOOL[plan.symmetric]),
        "  static constexpr int DIM = %d;" % plan.dim,
        "  static constexpr int NC = %d;" % plan.tangent_components,
        "  static constexpr bool symmetric = %s;" % _CPP_BOOL[plan.symmetric],
        "  static constexpr bool reads_state = %s;" % _CPP_BOOL[reads_state],
        "  static constexpr bool reads_previous = %s;" % _CPP_BOOL[reads_previous],
        "  static constexpr int n_parameters = %d;" % len(parameters),
        "  // The integration-point fields the tangent reads, and their components.",
        "  static constexpr int n_quadrature_fields = %d;" % len(fields),
    ]
    lines += _QUADRATURE_COMPONENT_LINES[bool(fields)](fields)
    lines += [
        "  static constexpr int max_quadrature_components = %d;" % max([c for _n, c in fields] + [1]),
        "",
        "  // The tangent at one point, unweighted: `gu_ref` (and `gz_ref`, the previous state's) is the",
        "  // reference gradient there, `adjugate` and `determinant` the geometry's.",
        "  template <typename s_t>",
        "  %s void point(const s_t *const RSTR parameters," % q,
        "                    const s_t *const RSTR gu_ref,",
        "                    const s_t *const RSTR gz_ref,",
        "                    const s_t *const RSTR adjugate,",
        "                    const s_t determinant,",
        "                    const s_t *const *const RSTR quadrature,",
        "                    s_t *const RSTR tangent) {",
    ]
    if not fields:
        lines.append("    (void)quadrature;")
    if not reads_state:
        lines.append("    (void)gu_ref;")
    if not reads_previous:
        lines.append("    (void)gz_ref;")
    if not parameters:
        lines.append("    (void)parameters;")
    lines += ["  " + line for line in body]
    lines += ["  }", "};", "", "}  // namespace codegen", "}  // namespace sfem", "", "#endif", ""]
    return struct, "\n".join(lines)


def inexact_unit_files(material, unit, context):
    """The stored-tangent files of one emission unit on one element, or ``()``.

    The unit is the one publishing the Jacobian action
    (`plans/residual_structure.publishes_inexact_apply`); its forms are summed
    into one tangent.
    """
    collection = getattr(unit, "form_collection", None)
    if collection is None:
        return ()
    dim = int(unit.dim)
    built = flux_form_for_collection(collection, dim)
    if built is None:
        return ()
    units = (built,)
    return inexact_files(
        str(getattr(unit, "name", None) or material.name),
        context.element_type,
        context.specialization.quadrature_rule,
        units,
        point_tangent_parameters(form for form, _ in units),
    )


def inexact_files(unit_name, element_type, rule, units, parameters):
    """`(path, source)` for one unit on one element, or ``()`` where there is no plan."""
    plans = inexact_plans(element_type, rule, [form for form, _ in units])
    return _FILES_BY_APPLICABILITY[plans is not None](unit_name, plans, units, parameters)


def device_store_factory(struct):
    """The declaration of a device unit's store factory, which its generated source defines."""
    return (
        "std::unique_ptr<InexactStore> %s_store(const ptrdiff_t nelements, "
        "const InexactOptions &options, void *const stream)" % struct
    )


def _device_store_source(struct):
    """The library's device stores, instantiated for one unit on one element.

    The operator is host code and cannot instantiate device templates, so each
    unit's stores are compiled here and reached through `<unit>_store`.
    """
    return "\n".join(
        [
            '#include "%s"' % current_target().header_name(struct),
            '#include "sfem_cuda_InexactStore.cuh"',
            "",
            "namespace sfem {",
            "namespace codegen {",
            "",
            "%s {" % device_store_factory(struct),
            "  return inexact::cuda::create_store<%s>(nelements, options, stream);" % struct,
            "}",
            "",
            "}  // namespace codegen",
            "}  // namespace sfem",
            "",
        ]
    )


#: A device unit's stores are compiled in a source of their own; the host's are
#: instantiated by the operator.
_STORE_SOURCES = {
    True: lambda struct: (
        ("%s_store.%s" % (struct, current_target().mesh_source_extension()), _device_store_source(struct)),
    ),
    False: lambda struct: (),
}


def _unit_files(unit_name, plans, units, parameters):
    element_plan, plan = plans
    struct, unit_source = unit_header_source(unit_name, element_plan, units, parameters)
    return (
        (current_target().header_name(struct), unit_source),
        (element_header_path(element_plan), element_header_source(element_plan)),
        (contract_header_path(plan.dim), contract_header_source(plan.dim)),
    ) + _STORE_SOURCES[current_target().kernel_vector_width() is None](struct)


#: Whether the plans cover this unit on this element decides whether there are files.
_FILES_BY_APPLICABILITY = {True: _unit_files, False: lambda *arguments: ()}
