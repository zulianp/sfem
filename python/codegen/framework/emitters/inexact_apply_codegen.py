"""The opt-in inexact apply, emitted as one header per element.

The plan is `plans/inexact_apply.py`; this prints it.

`Sbar` depends on the state and the geometry but not on the vector being
applied, so in a Krylov solve one tangent serves every apply of that Newton
step.  The header carries that split:

    <material>_<element>_inexact_apply_tangent_a_msoa
        once per tangent: state, geometry and material in, `Sbar` out

    <material>_<element>_inexact_apply_stored_a_msoa
    <material>_<element>_inexact_apply_compressed_a_msoa
        once per apply: `Sbar` and the vector in, nothing else

The apply kernels take no geometry, no state and no material parameters at all
-- the material has been evaluated away into `Sbar`, and what is left is a
contraction that is the same for every material.

`Sbar` is 45 numbers per element in three dimensions, whatever the element, so
the store is small and its precision is a free parameter: the kernels are
templated on the stored type.  The compressed apply adds one scale per element
and applies it to the *outputs*, of which there are `dim * n_nodes`, rather
than to the tangent's 45 components -- the action is linear in `Sbar`, so this
is the same number, arrived at with fewer multiplies and without a decompressed
copy of the tangent in registers.

Correctness is gated by the stored apply against the exact one: on an affine
simplex the projection loses nothing, so the two must agree to round-off, and
any difference there is a defect rather than an approximation.
"""

import dataclasses
import functools

import sympy as sp

from codegen.framework.emitters.cprinter import _sfem_ccode
from codegen.framework.emitters.kernel_prologue import kernel_constant
from codegen.framework.emitters.quadrature_codegen import (
    cpp_scalar_initializer_list,
    cpp_scalar_literal,
    quadrature_reference_accessor,
    reference_include_lines,
)
from codegen.framework.emitters.ast_printer import (
    CLikeKernelASTPrinter,
    render_kernel_ast_lines,
)
from codegen.framework.ir.kernel_ast import (
    FunctionDefNode,
    LoopKind,
    LoopNode,
    RawLinesNode,
    add_assign_increment,
    expr_ref,
    iteration_range,
    iterator,
    pre_increment,
)
from codegen.framework.plans.evaluation_strategy import (
    EvaluationStrategy,
    evaluation_strategy,
)
from codegen.framework.plans.inexact_apply import (
    action_stages,
    emittable_inexact_apply_plan,
    flux_form_for_collection,
    projected_tangent,
    quadrature_accumulation,
    reference_gradients_by_point,
)
from codegen.framework.targets import current_target

#: How the stored tangent is addressed.  The store is SoA: each component is its
#: own contiguous run over the elements, so the element index needs no stride at
#: all and one stride places the components.
_TANGENT_ADDRESS = "element + %d * tangent_component_stride"
#: The block's slice of the store, hoisted above the lane loop.  Being SoA is
#: what makes this a base pointer and nothing more: the component is contiguous
#: across the lanes, so it needs no staging and the lane indexes it directly.
_BLOCKED_TANGENT_BASE = "evb + %d * tangent_component_stride"
#: How the lane addresses it once the base pointer is in hand.
_BLOCKED_TANGENT_LANE = "btangent%d[lane]"


def inexact_apply_kernel_source(
    material_name,
    op_name,
    element_type,
    dim,
    n_nodes,
    flux_form,
    rule,
    parameter_names,
    is_deformation_gradient,
):
    """The header's source, or ``None`` when this path does not cover the case.

    Not covered: an element with no symbolic basis, a rule whose reference
    gradients are not the constants an affine simplex has, or a form whose flux
    does not differentiate into a tangent here.  The plan layer decides;
    emission looks the answer up.
    """
    plan = emittable_inexact_apply_plan(element_type, dim, n_nodes, flux_form, rule)
    return _KERNEL_BY_APPLICABILITY[plan is not None](
        plan,
        material_name,
        op_name,
        element_type,
        dim,
        n_nodes,
        flux_form,
        rule,
        parameter_names,
        is_deformation_gradient,
    )


#: The staging header, named only where a packed kernel stages through it.
_PACK_SCRATCH_INCLUDE = {
    True: ['#include "packed_thread_scratch.hpp"'],
    False: [],
}


def _emits_packed(plan):
    return any(layout != "standard" for layout in plan.apply_layouts)


def _inexact_apply_kernel_source(
    plan,
    material_name,
    op_name,
    element_type,
    dim,
    n_nodes,
    flux_form,
    rule,
    parameter_names,
    is_deformation_gradient,
):
    """The three kernels, reached only when the plan layer said they apply."""
    component = ["x", "y", "z"][:dim]
    adjugate = sp.Matrix(dim, dim, lambda r, c: sp.Symbol("adjugate%d" % (r * dim + c)))
    determinant = sp.Symbol("determinant")
    state = [
        [sp.Symbol("u%s_%d" % (component[c], j)) for j in range(n_nodes)]
        for c in range(dim)
    ]
    # A rate-dependent material also reads the previous state.  It is gathered
    # like the current one and absorbed into the tangent, so only the assembly
    # kernel grows an argument; the apply kernels are untouched.
    previous_state = [
        [sp.Symbol("z%s_%d" % (component[c], j)) for j in range(n_nodes)]
        for c in range(dim)
    ]
    increment = [
        [sp.Symbol("h%s_%d" % (component[c], j)) for j in range(n_nodes)]
        for c in range(dim)
    ]
    output = [
        [sp.Symbol("element_out%d_%d" % (c, p)) for p in range(n_nodes)]
        for c in range(dim)
    ]

    integrand, gradient_symbols = projected_tangent(
        plan,
        flux_form,
        rule,
        adjugate,
        determinant,
        state,
        is_deformation_gradient,
        previous_state,
    )
    quadrature_weights = quadrature_accumulation(rule)
    reference = reference_gradients_by_point(rule, n_nodes, dim)
    # Which state values the tangent actually reads.  A state-independent
    # material -- linear elasticity, whose tangent is constant -- reads none,
    # and then the kernel must not gather a state it will not use.  The plan
    # answers this; emission spells the answer.
    used_state = plan.state_dependence(integrand, state)
    used_previous = plan.state_dependence(integrand, previous_state)

    tangent_symbols = [
        sp.Symbol("tangent%d" % slot) for slot in range(plan.tangent_components)
    ]
    stages = action_stages(plan, tangent_symbols, increment, output)
    action_body = []
    for stage in stages:
        action_body.extend(_assignment_lines(stage.assignments, stage.name))

    parameters = tuple(str(name) for name in parameter_names)
    prefix = "%s_%s" % (material_name, str(element_type).lower())
    # The guard is not decoration: this header is included once today, but the
    # packed variant gave a second consumer a reason to include it and a second
    # inclusion redefines every kernel in it.
    lines = ["#pragma once", '#include "kernel_math.hpp"']
    # A packed kernel stages through thread-private scratch, which the framework
    # already publishes.  Named only when a packed layout is emitted, so a 2D
    # header does not include something it never calls.
    lines.extend(_PACK_SCRATCH_INCLUDE[_emits_packed(plan)])
    lines.extend(_REFERENCE_INCLUDES_BY_STRATEGY[evaluation_strategy(element_type)](rule, dim))
    lines.extend(["", "namespace sfem {", "namespace codegen {", ""])
    lines.extend(
        _tangent_lines(
            prefix, dim, n_nodes, component, parameters, used_state,
            used_previous, integrand, gradient_symbols, quadrature_weights,
            reference, rule, plan,
        )
    )
    # One per layout the plan names, in the order it names them.  Iterated, not
    # tested: `test_emission_is_a_printer` is the reason.
    for layout in plan.apply_layouts:
        lines.extend(
            _stored_lines(prefix, n_nodes, component, plan, action_body, layout)
        )
    lines.extend(_compressed_lines(prefix, n_nodes, component, plan, action_body))
    lines.extend(["} // namespace codegen", "} // namespace sfem", ""])

    # The extern "C" definitions go in their own translation unit, the way the
    # rest of the generated operators are laid out: the header carries the
    # templates, the source carries the symbols the library links against.
    # The c_abi header first, exactly as the other generated operator sources
    # do it: it is what brings in idx_t, geom_t, SFEM_RESTRICT and the storage
    # types.  Including only the inline header leaves the kernel signatures
    # unparseable, which a spike driver hides by including sfem_base.hpp itself
    # before it -- and which the library build does not.
    operator_lines = [
        '#include "../../op/sfem_%s_c_abi.hpp"' % op_name,
        "",
        '#include "%s_inexact_apply_inline.hpp"' % prefix,
        "",
    ]
    operator_lines.extend(
        _c_abi_lines(
            prefix, dim, n_nodes, component, parameters, used_previous,
            bool(used_state),
            tuple(l for l in plan.apply_layouts if l != "standard"),
        )
    )
    return (
        "%s_inexact_apply_tangent_a_msoa" % prefix,
        "\n".join(lines),
        "\n".join(operator_lines),
    )


#: Whether the plan layer produced a plan decides whether there is a kernel.
#: A table, for the reason the energy emitter's bodies are ones: emission looks
#: the answer up rather than deciding it again.
_KERNEL_BY_APPLICABILITY = {
    True: lambda plan, *arguments: _inexact_apply_kernel_source(plan, *arguments),
    False: lambda plan, *arguments: None,
}


def _target_pragma(method, *arguments):
    """One pragma from the bound target, or nothing if it has none.

    Every pragma this emitter prints goes through here; `test_target_binding`
    gates that, and the reason is that a hardcoded `#pragma omp` is a kernel that
    silently means something else on a target that is not OpenMP.
    """
    target = current_target()
    if target is None or not hasattr(target, method):
        return []
    pragma = getattr(target, method)(*arguments)
    return [pragma] if pragma else []


def _parallel_loop_lines():
    """The pragma opening the element loop, from the bound target."""
    target = current_target()
    if target is None or not hasattr(target, "parallel_element_loop_lines"):
        return []
    return ["  %s" % line for line in target.parallel_element_loop_lines("static")]


def _scatter_lines(lhs, rhs, indent):
    """A scatter-add spelled by the bound target rather than by a literal."""
    target = current_target()
    if target is None or not hasattr(target, "scatter_add_lines"):
        return ["%s%s += %s;" % (indent, lhs, rhs)]
    return list(target.scatter_add_lines(lhs, rhs, indent))


def _assignment_lines(assignments, prefix, indent="    "):
    """Common subexpressions first, then the named values, as C declarations."""
    if not assignments:
        return []
    symbols = [symbol for symbol, _expression in assignments]
    expressions = [expression for _symbol, expression in assignments]
    temporaries, reduced = sp.cse(
        expressions, symbols=sp.numbered_symbols("%s_t" % prefix)
    )
    lines = [
        "%sconst s_t %s = %s;" % (indent, symbol, _sfem_ccode(expression))
        for symbol, expression in temporaries
    ]
    lines.extend(
        "%sconst s_t %s = %s;" % (indent, symbol, _sfem_ccode(expression))
        for symbol, expression in zip(symbols, reduced)
    )
    return lines


def _element_lines(n_nodes, indent="    "):
    return [
        "%sconst idx_t ev%d = elements[%d][element];" % (indent, node, node)
        for node in range(n_nodes)
    ]


def _gather_lines(role, component, n_nodes, wanted, indent="    "):
    """Element gathers for one role, restricted to the values that are read."""
    return [
        "%sconst s_t %s%s_%d = %s%s[ev%d * %s_stride];"
        % (indent, role, name, node, role, name, node, role)
        for name in component
        for node in range(n_nodes)
        if sp.Symbol("%s%s_%d" % (role, name, node)) in wanted
    ]


def _geometry_lines(dim, indent="    "):
    lines = [
        "%sconst s_t adjugate%d = s_t(g_adj%d[element]);"
        % (indent, index, index)
        for index in range(dim * dim)
    ]
    lines.append(
        "%sconst s_t determinant = s_t(g_det0[element]);"
        % indent
    )
    return lines


def _geometry_arguments(dim):
    lines = [
        "    const g_t *const RSTR g_adj%d," % index
        for index in range(dim * dim)
    ]
    lines.append("    const g_t *const RSTR g_det0,")
    return lines


def _stream_arguments(role, component):
    lines = ["    const ptrdiff_t %s_stride," % role]
    lines.extend(
        "    const s_t *const RSTR %s%s," % (role, name)
        for name in component
    )
    return lines


def _output_arguments(component):
    lines = ["    const ptrdiff_t out_stride,"]
    lines.extend(
        "    s_t *const RSTR out%s%s"
        % (name, "," if index + 1 < len(component) else "")
        for index, name in enumerate(component)
    )
    return lines


def _scatter_body(component, n_nodes, scale=""):
    lines = []
    for index, name in enumerate(component):
        for node in range(n_nodes):
            lines.extend(
                _scatter_lines(
                    "out%s[ev%d * out_stride]" % (name, node),
                    "%selement_out%d_%d" % (scale, index, node),
                    "    ",
                )
            )
    return lines


def _gathered_names(role, component, n_nodes, wanted):
    """The values one role contributes, as (name, source, node, role) tuples."""
    return [
        ("%s%s_%d" % (role, name, node), "%s%s" % (role, name), node, role)
        for name in component
        for node in range(n_nodes)
        if sp.Symbol("%s%s_%d" % (role, name, node)) in wanted
    ]


@dataclasses.dataclass(frozen=True)
class _ReferenceTable:
    """What `reference_include_lines` asks of a table: its name."""

    name: str


def _shared_reference_includes(rule, dim):
    """The shared headers the quadrature loop forwards into."""
    references = [_ReferenceTable("grad_ref_%s" % axis) for axis in _REFERENCE_AXES[:dim]]
    references.append(_ReferenceTable("q_weight"))
    return list(reference_include_lines(rule, references))


def _no_reference_includes(rule, dim):
    """A kernel that reads no shared table includes none."""
    return []


#: Only the strategy that reads the shared structs includes them.
_REFERENCE_INCLUDES_BY_STRATEGY = {
    EvaluationStrategy.QUADRATURE: _shared_reference_includes,
    EvaluationStrategy.EXPANDED: _no_reference_includes,
    EvaluationStrategy.SUM_FACTORIZED: _no_reference_includes,
}


def _quadrature_table_lines(name, values, indent="    "):
    """One rule constant per point, as a folded table the loop indexes.

    Literals rather than a runtime argument: these are properties of the element
    and the rule, settled before the kernel exists.  Evaluated here rather than
    printed symbolically, because a Gauss rule's gradients carry `sqrt(3)` and
    `sqrt` is not constexpr -- printing the expression would not compile, and
    printing it into a non-constexpr table would put a square root in the
    kernel's prologue for a number the generator already knows.  The same
    seventeen digits the generated reference tables use.
    """
    return [
        "%sstatic constexpr s_t %s[%d] = {%s};"
        % (
            indent,
            name,
            len(values),
            cpp_scalar_initializer_list(
                [float(sp.sympify(value).evalf(20)) for value in values]
            ),
        )
    ]


def _quadrature_loop_compute(
    integrand, integrand_symbols, gradient_symbols, reference, weights,
    n_nodes, dim, rule, plan, strategy=None,
):
    """The sum over several points: accumulators, then the loop.

    Reads the shared reference struct the rule publishes -- the same tables
    every other kernel on this element reads -- rather than a table of its own.
    """
    lines = [
        "      s_t tangent%d = s_t(0);" % slot
        for slot in range(plan.tangent_components)
    ]
    lines.append("      for (int q = 0; q < NQ; ++q) {")
    lines.extend(
        "        const s_t %s = %s;"
        % (
            gradient_symbols[node][axis],
            _GRADIENT_READ_BY_STRATEGY[strategy](node, axis, n_nodes, dim),
        )
        for node in range(n_nodes)
        for axis in range(dim)
    )
    lines.append("        const s_t qw = %s;" % _WEIGHT_READ_BY_STRATEGY[strategy])
    lines.extend(
        "    %s" % line
        for line in _assignment_lines(
            list(zip(integrand_symbols, integrand)), "integrand", indent="        "
        )
    )
    lines.extend(
        "        tangent%d += qw * integrand%d;" % (slot, slot)
        for slot in range(plan.tangent_components)
    )
    lines.append("      }")
    return lines


def _single_point_compute(
    integrand, integrand_symbols, gradient_symbols, reference, weights,
    n_nodes, dim, rule, plan,
):
    """An EXPANDED element: no loop, no tables, the gradients folded in.

    `plans.evaluation_strategy` says a lowest-order simplex evaluates in closed
    form -- "no quadrature loop, no per-point geometry, no reference-basis
    tables" -- and this is that, for the tangent.
    """
    substitution = {
        gradient_symbols[node][axis]: reference[0][node][axis]
        for node in range(n_nodes)
        for axis in range(dim)
    }
    weight = weights[0]
    folded = [weight * expression.xreplace(substitution) for expression in integrand]
    tangent_symbols = [
        sp.Symbol("tangent%d" % slot) for slot in range(plan.tangent_components)
    ]
    return [
        "  %s" % line
        for line in _assignment_lines(list(zip(tangent_symbols, folded)), "tangent")
    ]


def _folded_gradient_tables(reference, weights, n_nodes, dim, rule):
    values = [
        reference[point][node][axis]
        for point in range(len(weights))
        for node in range(n_nodes)
        for axis in range(dim)
    ]
    return (
        [kernel_constant("NQ", len(weights), indent="    ")]
        + _quadrature_table_lines("QGRAD", values)
        + _quadrature_table_lines("QWEIGHT", weights)
    )


def _no_quadrature_tables(reference, weights, n_nodes, dim, rule):
    """An EXPANDED element indexes no table, so it declares none."""
    return []


def _shared_reference_tables(reference, weights, n_nodes, dim, rule):
    """Aliases onto the shared reference struct this rule already publishes.

    `ref_<element>_<rule><s_t>::grad_ref_x()` and `quad_<cell>_<rule>::q_weight()`
    are what every other kernel on this element reads; a private copy is what
    `quadrature_reference_accessor` was written to remove.
    """
    lines = [kernel_constant("NQ", len(weights), indent="    ")]
    lines.extend(
        "    const s_t *const RSTR qgrad_%s = %s;"
        % (axis, quadrature_reference_accessor(rule, "grad_ref_%s" % axis))
        for axis in _REFERENCE_AXES[:dim]
    )
    lines.append(
        "    const s_t *const RSTR qweight = %s;"
        % quadrature_reference_accessor(rule, "q_weight")
    )
    # The shared table holds the rule's own weights; `Sbar` is their weighted
    # average, so the reciprocal of their sum rides beside them as one constant.
    measure = sum((sp.nsimplify(weight) for weight in rule.weights), sp.Integer(0))
    lines.append(
        "    static constexpr s_t QMEASURE = %s;"
        % cpp_scalar_literal(float(sp.Integer(1) / measure))
    )
    return lines


#: The reference gradient tables are named by axis, as the shared structs name
#: them.
_REFERENCE_AXES = ("x", "y", "z")


#: Keyed on whether the rule has a single point, so emission looks the answer up
#: rather than branching on the rule.
#: How the tangent's quadrature sum is emitted, per evaluation strategy.  A
#: table, because the choice belongs to `plans.evaluation_strategy` and emission
#: only looks up the answer.
#:
#: SUM_FACTORIZED is honest rather than correct here: the strategy calls for
#: contraction through one-dimensional operators and this kernel still walks the
#: points with the element's full reference gradients, which is why it carries a
#: folded gradient table instead of reading the shared 1D factors.  That is the
#: remaining gap, and naming it in this table is what keeps it visible.
_QUADRATURE_BY_STRATEGY = {
    EvaluationStrategy.EXPANDED: _single_point_compute,
    EvaluationStrategy.QUADRATURE: functools.partial(
        _quadrature_loop_compute, strategy=EvaluationStrategy.QUADRATURE
    ),
    EvaluationStrategy.SUM_FACTORIZED: functools.partial(
        _quadrature_loop_compute, strategy=EvaluationStrategy.SUM_FACTORIZED
    ),
}

_QUADRATURE_TABLES_BY_STRATEGY = {
    EvaluationStrategy.EXPANDED: _no_quadrature_tables,
    EvaluationStrategy.QUADRATURE: _shared_reference_tables,
    EvaluationStrategy.SUM_FACTORIZED: _folded_gradient_tables,
}

#: Where one point's reference gradient is read from, per strategy.
_GRADIENT_READ_BY_STRATEGY = {
    EvaluationStrategy.QUADRATURE: (
        lambda node, axis, n_nodes, dim: "qgrad_%s[q * %d + %d]"
        % (_REFERENCE_AXES[axis], n_nodes, node)
    ),
    EvaluationStrategy.SUM_FACTORIZED: (
        lambda node, axis, n_nodes, dim: "QGRAD[q * %d + %d]"
        % (n_nodes * dim, node * dim + axis)
    ),
}

_WEIGHT_READ_BY_STRATEGY = {
    EvaluationStrategy.QUADRATURE: "qweight[q] * QMEASURE",
    EvaluationStrategy.SUM_FACTORIZED: "QWEIGHT[q]",
}


def _tangent_lines(
    prefix, dim, n_nodes, component, parameters, used_state, used_previous,
    integrand, gradient_symbols, weights, reference, rule, plan,
):
    """The partial assembly: `Sbar` accumulated over the quadrature points.

    Blocked over `VS` elements like the applies: the state reaches this kernel
    through the connectivity and so arrives indirect, and staging it lane-major
    first is what lets the arithmetic that follows vectorise.

    The quadrature sum is a loop here, as it is in every other kernel in this
    framework.  It used to be carried out symbolically in the plan, which
    inlined the whole tangent once per point -- eight copies for a HEX8 rule --
    and produced a single basic block of 3855 statements that cost gcc 234.8 s
    and 4.45 GB on one kernel.  The arithmetic is the same; what changed is that
    the repetition is a loop again.

    Takes the previous state only when the material reads one, so a
    rate-independent material keeps the shorter signature.
    """
    tangent_symbols = [
        sp.Symbol("tangent%d" % slot) for slot in range(plan.tangent_components)
    ]
    integrand_symbols = [
        sp.Symbol("integrand%d" % slot) for slot in range(plan.tangent_components)
    ]
    gathered = _gathered_names("u", component, n_nodes, used_state)
    gathered.extend(_gathered_names("z", component, n_nodes, used_previous))

    # Only a material whose tangent reads the state needs the connectivity: a
    # linear one is a function of the geometry alone, and gathering nodes it
    # never looks at would leave an empty lane loop behind.
    scratch = _CONNECTIVITY_BY_USE[bool(gathered)](n_nodes)
    scratch.extend("    s_t b%s[VS];" % value for value, _s, _n, _r in gathered)

    gathers = _STAGED_GATHERS_BY_USE[bool(gathered)](gathered)
    gathers.extend(_blocked_geometry_bases(dim))
    gathers.extend(_blocked_tangent_bases(plan.tangent_components))

    # A one-point rule is a sum with one term.  Opening a loop for it, and
    # reading the gradients out of a table the loop indexes, would be arithmetic
    # the generator already knows the answer to -- and the affine simplices,
    # where the projection is exact, are all one-point rules.  They get the
    # point's gradients folded in as the literals they are.
    # How this element is evaluated is `plans.evaluation_strategy`'s answer, not
    # a property of the rule this file re-derives.  A one-point rule and an
    # EXPANDED element coincide on the simplices, which is why testing the point
    # count looked right; it is the element's family that decides.
    strategy = evaluation_strategy(plan.element_type)
    compute = _QUADRATURE_BY_STRATEGY[strategy](
        integrand, integrand_symbols, gradient_symbols, reference, weights,
        n_nodes, dim, rule, plan,
    )
    compute.extend(
        "      %s = tangent_t(tangent%d);" % (_BLOCKED_TANGENT_LANE % slot, slot)
        for slot in range(plan.tangent_components)
    )
    # The per-lane state and geometry are read once, before the quadrature loop:
    # they do not depend on the point.
    compute = [
        "      const s_t %s = b%s[lane];" % (value, value)
        for value, _s, _n, _r in gathered
    ] + _blocked_geometry_lines(dim) + compute

    signature = ["    const ptrdiff_t nelements,"]
    signature.extend(_CONNECTIVITY_ARGUMENT_BY_USE[bool(gathered)])
    signature.extend(_geometry_arguments(dim))
    signature.extend("    const s_t %s," % name for name in parameters)
    signature.extend(_STATE_STREAMS_BY_USE[bool(used_state)](component))
    signature.extend(_PREVIOUS_STREAMS_BY_USE[bool(used_previous)](component))
    signature.extend(
        [
            "    const ptrdiff_t tangent_component_stride,",
            "    tangent_t *const RSTR tangent",
        ]
    )
    tables = _QUADRATURE_TABLES_BY_STRATEGY[strategy](
        reference, weights, n_nodes, dim, rule
    )
    return _blocked_function_lines(
        "%s_inexact_apply_tangent_a_msoa" % prefix,
        ("typename s_t", "typename g_t", "typename tangent_t", "int VS"),
        signature, tables + scratch, gathers, compute, [],
    )


def _stored_lines(prefix, n_nodes, component, plan, action_body, layout="standard"):
    """The apply: stored tangent and the vector, and nothing else.

    Two layouts, one arithmetic.  Everything between the gather and the scatter is
    built once and handed to both, because it *is* the same computation -- what a
    packed mesh changes is where the increment is read from and where the output
    is accumulated, and nothing else.  The layout is chosen by the caller from the
    plan and looked up here; emission does not decide it.
    """
    gathered = _gathered_names(
        "h", component, n_nodes, _all_names("h", component, n_nodes)
    )
    scratch = _connectivity_scratch(n_nodes, _CONNECTIVITY_INDEX_TYPE[layout])
    scratch.extend("    s_t b%s[VS];" % value for value, _s, _n, _r in gathered)
    scratch.extend(
        "    s_t bout%d_%d[VS];" % (index, node)
        for index in range(len(component))
        for node in range(n_nodes)
    )
    # Staged outside the arithmetic loop, one pass each: these are indirect and
    # will not vectorise, and leaving them inside stops the arithmetic
    # vectorising with them.  The tangent is not staged -- component-major makes
    # it contiguous across the lanes already, and it stays so under packing
    # because packs are contiguous ranges of elements.
    gathers = _STORED_GATHERS_BY_LAYOUT[layout](gathered, component)
    gathers.extend(_blocked_tangent_bases(plan.tangent_components, "const "))
    compute = [
        "      const s_t %s = b%s[lane];" % (value, value)
        for value, _s, _n, _r in gathered
    ]
    compute.extend(
        "      const s_t tangent%d = s_t(%s);" % (slot, _BLOCKED_TANGENT_LANE % slot)
        for slot in range(plan.tangent_components)
    )
    compute.extend("  %s" % line for line in action_body)
    compute.extend(
        "      bout%d_%d[lane] = element_out%d_%d;" % (index, node, index, node)
        for index in range(len(component))
        for node in range(n_nodes)
    )
    store = _STORED_SCATTER_BY_LAYOUT[layout](component, n_nodes)

    signature = _STORED_PROLOGUE_BY_LAYOUT[layout](component, plan.tangent_components)
    signature.extend(_stream_arguments("h", component))
    signature.extend(_output_arguments(component))
    return _STORED_SKELETON_BY_LAYOUT[layout](
        "%s_inexact_apply_stored%s_a_msoa" % (prefix, _LAYOUT_SUFFIX[layout]),
        ("typename s_t", "typename tangent_t", "int VS"),
        signature, scratch, gathers, compute, store, component,
    )


def _standard_stored_prologue(_component, _n_components):
    return [
        "    const ptrdiff_t nelements,",
        "    idx_t **const RSTR elements,",
        "    const ptrdiff_t tangent_component_stride,",
        "    const tangent_t *const RSTR tangent,",
    ]


def _compressed_lines(prefix, n_nodes, component, plan, action_body):
    """The same apply from a scaled low-precision store.

    The scale multiplies the outputs, not the tangent: the action is linear in
    `Sbar`, so it is the same number either way, and there are `dim * n_nodes`
    outputs against the tangent's 45 components.
    """
    body = _element_lines(n_nodes)
    body.extend(_gather_lines("h", component, n_nodes, _all_names("h", component, n_nodes)))
    body.append("    const s_t scale = s_t(scaling[element]);")
    body.extend(
        "    const s_t tangent%d = s_t(tangent[%s]);"
        % (slot, _TANGENT_ADDRESS % slot)
        for slot in range(plan.tangent_components)
    )
    body.extend(action_body)
    body.extend(_scatter_body(component, n_nodes, scale="scale * "))

    signature = ["    const ptrdiff_t nelements,", "    idx_t **const RSTR elements,"]
    signature.extend(
        [
            "    const ptrdiff_t tangent_component_stride,",
            "    const tangent_t *const RSTR tangent,",
            "    const scale_t *const RSTR scaling,",
        ]
    )
    signature.extend(_stream_arguments("h", component))
    signature.extend(_output_arguments(component))
    return _function_lines(
        "%s_inexact_apply_compressed_a_msoa" % prefix,
        ("typename s_t", "typename tangent_t", "typename scale_t"),
        signature,
        body,
    )


#: Whether the material reads a previous state decides whether the assembly
#: kernel takes one.  A table rather than a branch, for the reason the rest of
#: this emitter is: the decision belongs to the plan, which already made it in
#: `state_dependence`, and emission spells the answer.
_PREVIOUS_STREAMS_BY_USE = {
    True: lambda component: _stream_arguments("z", component),
    False: lambda component: [],
}
#: And whether it reads the current one decides the same for that, for the same
#: reason: linear elasticity's tangent is a function of the geometry alone.
_STATE_STREAMS_BY_USE = {
    True: lambda component: _stream_arguments("u", component),
    False: lambda component: [],
}
#: The connectivity is there to be gathered through.  A kernel that gathers
#: nothing does not take it.
_CONNECTIVITY_ARGUMENT_BY_USE = {
    True: ["    idx_t **const RSTR elements,"],
    False: [],
}
#: The `extern "C"` wrapper takes what it forwards, and no more.  Leaving the
#: parameter in place unnamed was the other option and is not available: the
#: dispatch layer builds its call out of the published signature's parameter
#: *names*, so a nameless parameter there produces a call with no argument for
#: it.  The published signature varies with what the material reads, exactly as
#: it already does for a previous state.
_CONNECTIVITY_CALL_BY_USE = {True: ["elements"], False: []}
_CONNECTIVITY_ABI_BY_USE = {
    True: ["    idx_t **const RSTR elements,"],
    False: [],
}
_ABI_STATE_BY_USE = {
    True: lambda scalar, component: _abi_stream(scalar, "u", component),
    False: lambda scalar, component: [],
}
_STATE_CALL_BY_USE = {
    True: lambda component: ["      u_stride, %s," % ", ".join(
        "u%s" % name for name in component
    )],
    False: lambda component: [],
}


def _all_names(role, component, n_nodes):
    return frozenset(
        sp.Symbol("%s%s_%d" % (role, name, node))
        for name in component
        for node in range(n_nodes)
    )


def _simd_pragma(indent):
    """The lane loop's vectorize pragma, from the bound target.

    Spelled by the target rather than here, like the element loop above it: a
    backend that does not vectorise this way returns nothing and the loop is
    emitted plain.
    """
    target = current_target()
    if target is None or not hasattr(target, "vectorize_pragma"):
        return []
    pragma = target.vectorize_pragma()
    return ["%s%s" % (indent, pragma)] if pragma else []


def _lane_loop(body, indent="    "):
    return _simd_pragma(indent) + [
        "%sfor (int lane = 0; lane < ne; ++lane) {" % indent, *body, "%s}" % indent,
    ]


def _connectivity_scratch(n_nodes, index_type="idx_t"):
    """The block's connectivity, gathered in one pass.

    One loop with `n_nodes` statements rather than `n_nodes` loops with one
    each: the stores belong in a single SIMD region, and opening one per node
    is the pattern `tests/test_kernels_are_lean.py` gates against.

    A packed mesh numbers its nodes within the pack, so its connectivity is
    `uint16_t` rather than `idx_t` -- the index type is the layout's, not this
    function's.
    """
    scratch = ["    %s bev%d[VS];" % (index_type, node) for node in range(n_nodes)]
    scratch.extend(
        _lane_loop(
            [
                "      bev%d[lane] = elements[%d][evb + lane];" % (node, node)
                for node in range(n_nodes)
            ]
        )
    )
    return scratch


def _blocked_geometry_bases(dim):
    """The block's geometry as base pointers, one per component.

    The geometry is element-indexed and therefore already contiguous across the
    block, so it needs no staging -- only a pointer to the block's first
    element, so the lane loop reads `bg_adj0[lane]` rather than rebuilding the
    address per lane.
    """
    lines = [
        "    const g_t *const RSTR bg_adj%d = g_adj%d + evb;" % (index, index)
        for index in range(dim * dim)
    ]
    lines.append("    const g_t *const RSTR bg_det0 = g_det0 + evb;")
    return lines


def _blocked_geometry_lines(dim, indent="      "):
    lines = [
        "%sconst s_t adjugate%d = s_t(bg_adj%d[lane]);" % (indent, index, index)
        for index in range(dim * dim)
    ]
    lines.append("%sconst s_t determinant = s_t(bg_det0[lane]);" % indent)
    return lines


def _blocked_tangent_bases(n_components, qualifier=""):
    """The block's slice of the store, one base pointer per component.

    The two strides stay in the address because the ABI carries both layouts;
    what leaves the loop is everything that does not depend on the lane.
    """
    return [
        "    %stangent_t *const RSTR btangent%d = tangent + %s;"
        % (qualifier, slot, _BLOCKED_TANGENT_BASE % slot)
        for slot in range(n_components)
    ]


def _staged_gathers(gathered):
    """The block's indirect reads, staged lane-major in one pass."""
    return _lane_loop(
        [
            "      b%s[lane] = %s[bev%d[lane] * %s_stride];"
            % (value, source, node, role)
            for value, source, node, role in gathered
        ]
    )


#: A kernel that gathers nothing needs neither the pass nor the connectivity it
#: would read through.  Tables rather than branches, as elsewhere here.
_STAGED_GATHERS_BY_USE = {True: _staged_gathers, False: lambda gathered: []}
_CONNECTIVITY_BY_USE = {
    True: lambda n_nodes: _connectivity_scratch(n_nodes),
    False: lambda n_nodes: [],
}


#: How a packed kernel names its thread-private staging.  `conventions.py` already
#: reserves the `pk_` prefix for it.
_PACK_SCRATCH = "pk_%s"


def _packed_staged_gathers(gathered, component):
    """The block's reads, from pack-local scratch instead of global memory.

    The index is the same `bev` the standard kernel gathers, except that it is a
    pack-local slot rather than a global node, so this is a read from an array a
    few tens of kilobytes wide that this thread alone owns.  That is the whole of
    what packing buys on the gather side.
    """
    return _lane_loop(
        [
            "      b%s[lane] = pk_h[%d * max_nodes_per_pack + bev%d[lane]];"
            % (value, component.index(source[-1]), node)
            for value, source, node, _role in gathered
        ]
    )


def _packed_scatter(component, n_nodes):
    """Scatter into the pack's own scratch -- no atomic, because nothing shares it.

    Still serial over the lanes, for the reason the standard scatter is: two
    lanes of one block can land on the same node.  What is gone is the
    `#pragma omp atomic update`, and with it the contended global read-modify-write
    that this kernel spent most of its time on.
    """
    lines = []
    for index, _name in enumerate(component):
        for node in range(n_nodes):
            lines.append("    for (int lane = 0; lane < ne; ++lane) {")
            lines.append(
                "      pk_out[%d * max_nodes_per_pack + bev%d[lane]] += bout%d_%d[lane];"
                % (index, node, index, node)
            )
            lines.append("    }")
    return lines


def _packed_signature(component, n_components):
    """The packed ABI's prologue, in the order every packed kernel takes it.

    `n_shared_nodes` is absent: it exists so a one-pass kernel can tell which
    owned nodes need an atomic, and a two-pass kernel never scatters atomically
    at all.
    """
    return [
        "    const ptrdiff_t n_packs,",
        "    const ptrdiff_t n_elements_per_pack,",
        "    const ptrdiff_t nelements,",
        "    const ptrdiff_t max_nodes_per_pack,",
        "    uint16_t **const RSTR elements,",
        "    const ptrdiff_t *const RSTR owned_nodes_ptr,",
        "    const ptrdiff_t n_ghost_entries,",
        "    const ptrdiff_t n_ghost_reduce_rows,",
        "    const ptrdiff_t *const RSTR ghost_ptr,",
        "    const idx_t *const RSTR ghost_idx,",
        "    const ptrdiff_t *const RSTR ghost_reduce_ptr,",
        "    const ptrdiff_t *const RSTR ghost_reduce_idx,",
        "    const idx_t *const RSTR ghost_reduce_dest,",
        "    s_t *const RSTR ghost_buf,",
        "    const ptrdiff_t tangent_component_stride,",
        "    const tangent_t *const RSTR tangent,",
    ]


def _packed_function_lines(name, template_params, signature, scratch, gathers, compute,
                           store, component):
    """One kernel over a packed mesh, two-pass.

    The element's arithmetic is the standard kernel's, unchanged and in the same
    place; what differs is on either side of it.  The gather runs once per *pack*
    into thread-private scratch instead of once per element through the mesh
    connectivity, the scatter accumulates into that scratch without an atomic,
    and only the pack boundary reaches global memory -- the ghosts through a
    buffer that the disjoint second pass below reduces.
    """
    n_components = len(component)
    lines = ["  static constexpr int NC = %d;" % n_components]
    lines.append(
        "  const s_t *const h_components[NC] = {%s};"
        % ", ".join("h%s" % name for name in component)
    )
    lines.append(
        "  s_t *const out_components[NC] = {%s};"
        % ", ".join("out%s" % name for name in component)
    )
    lines.extend([""])
    lines.extend("  %s" % line for line in _target_pragma("parallel_region_pragma"))
    lines.extend(
        [
            "  {",
            "    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>("
            "2, (size_t)NC * (size_t)max_nodes_per_pack);",
            "    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>("
            "3, (size_t)NC * (size_t)max_nodes_per_pack);",
        ]
    )
    lines.extend("    %s" % line
                 for line in _target_pragma("worksharing_for_pragma", "static"))
    lines.extend(
        [
            "    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
            "      const ptrdiff_t e_start = pack * n_elements_per_pack;",
            "      const ptrdiff_t e_end = (nelements < (pack + 1) * n_elements_per_pack)",
            "                                  ? nelements",
            "                                  : (pack + 1) * n_elements_per_pack;",
            "      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
            "      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
            "      const ptrdiff_t ghost_off = ghost_ptr[pack];",
            "      const idx_t *const RSTR ghosts = &ghost_idx[ghost_off];",
            "",
            "      for (int d = 0; d < NC; ++d) {",
            "        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;",
            "        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;",
            "        const s_t *const RSTR h_component = h_components[d];",
            "        for (ptrdiff_t k = 0; k < n_contiguous + n_ghost; ++k) {",
            "          pk_component_out[k] = s_t(0);",
            "        }",
            "        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
            "          pk_h_component[k] = h_component[(owned_nodes_ptr[pack] + k) * h_stride];",
            "        }",
            "        for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
            "          pk_h_component[n_contiguous + k] = h_component[ghosts[k] * h_stride];",
            "        }",
            "      }",
            "",
            "      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {",
            "        const int ne = (int)((e_end - evb) < (ptrdiff_t)VS "
            "? (e_end - evb) : (ptrdiff_t)VS);",
        ]
    )
    for block in (scratch, gathers):
        lines.extend("    %s" % line for line in block)
    lines.extend(_simd_pragma("        "))
    lines.append("        for (int lane = 0; lane < ne; ++lane) {")
    lines.extend("    %s" % line for line in compute)
    lines.append("        }")
    lines.extend("    %s" % line for line in store)
    lines.append("      }")
    lines.extend(
        [
            "",
            "      for (int d = 0; d < NC; ++d) {",
            "        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;",
            "        s_t *const RSTR global_out = out_components[d];",
            "        s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;",
            "        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
            "          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];",
            "        }",
            "        for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
            "          ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];",
            "        }",
            "      }",
            "    }",
            "  }",
            "",
        ]
    )
    lines.extend("  %s" % line
                 for line in _target_pragma("parallel_for_pragma", "static"))
    lines.extend(
        [
            "  for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {",
            "    const idx_t dest = ghost_reduce_dest[row];",
            "    const ptrdiff_t begin = ghost_reduce_ptr[row];",
            "    const ptrdiff_t end = ghost_reduce_ptr[row + 1];",
            "    for (int d = 0; d < NC; ++d) {",
            "      const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;",
            "      s_t sum = s_t(0);",
            "      for (ptrdiff_t j = begin; j < end; ++j) {",
            "        sum += ghost_component[ghost_reduce_idx[j]];",
            "      }",
            "      out_components[d][dest * out_stride] += sum;",
            "    }",
            "  }",
            "  return SFEM_SUCCESS;",
        ]
    )
    # The packed kernel's two passes are still text, but it is a
    # `FunctionDefNode` like every other kernel here, so the signature, the
    # qualifier and the template line come from the node and the printer rather
    # than from string concatenation in this file.
    nodes = (
        FunctionDefNode(
            name="%s_impl" % name,
            params=tuple(line.strip().rstrip(",") for line in signature),
            body=(RawLinesNode(tuple(lines), reason="packed two-pass body"),),
            return_type="int",
            qualifier="static SFEM_INLINE",
            template_params=tuple(template_params),
        ),
    )
    return list(render_kernel_ast_lines(name, nodes)) + [""]


def _blocked_function_lines(name, template_params, signature, scratch, gathers, compute, store):
    """One kernel, blocked over `VS` elements, with the gathers staged.

    The values this kernel reads come through the mesh connectivity, so their
    loads are indirect and cannot vectorise.  They are staged into lane-major
    scratch in their own passes, which is how the exact apply in this framework
    gets vector code out of the arithmetic that follows: by the time the
    arithmetic loop runs, everything it touches is contiguous in the lane.

    Built as a `FunctionDefNode` and rendered through the IR printer, like every
    other kernel in the tree.  This path used to concatenate strings from the
    emitter straight to a file, which is why no pass, ratchet or printer setting
    ever applied to it and why it drifted into shapes nothing else could produce.
    The arithmetic bodies are still `RawLinesNode`, which is the escape hatch the
    IR documents for exactly this: the kernel is a tree now, and what is left as
    text is marked and countable.
    """
    target = current_target()
    pragma = target.vectorize_pragma() if hasattr(target, "vectorize_pragma") else None
    printer = CLikeKernelASTPrinter(vectorize_pragma=pragma or "")

    element_block = iterator("evb", "ptrdiff_t")
    lane = iterator("lane", "int")
    body = [
        RawLinesNode(
            (
                "    const int ne = (int)((nelements - evb) < (ptrdiff_t)VS "
                "? (nelements - evb) : (ptrdiff_t)VS);",
            ),
            reason="tile extent",
        ),
        RawLinesNode(tuple(scratch), reason="staging buffers"),
        RawLinesNode(tuple(gathers), reason="indirect gathers"),
        LoopNode(
            LoopKind.SIMD,
            lane,
            iteration_range(0, expr_ref("ne", "tile_extent")),
            pre_increment(lane),
            body=(RawLinesNode(tuple(compute), reason="kernel arithmetic"),),
            vectorized=bool(pragma),
        ),
        RawLinesNode(tuple(store), reason="scatter"),
    ]
    nodes = (
        FunctionDefNode(
            name="%s_impl" % name,
            params=tuple(line.strip().rstrip(",") for line in signature),
            body=(
                RawLinesNode(tuple(_parallel_loop_lines()), reason="element loop pragma"),
                LoopNode(
                    LoopKind.KERNEL,
                    element_block,
                    iteration_range(0, expr_ref("nelements", "mesh_extent")),
                    add_assign_increment(element_block, expr_ref("VS", "tile_width")),
                    body=tuple(body),
                ),
                RawLinesNode(("", "  return SFEM_SUCCESS;"), reason="status"),
            ),
            return_type="int",
            qualifier="static SFEM_INLINE",
            template_params=tuple(template_params),
        ),
    )
    return list(render_kernel_ast_lines(name, nodes, printer=printer)) + [""]


def _blocked_scatter(component, n_nodes, scale=""):
    """Scatter the block's outputs, serially over the lanes.

    Not a lane loop: two lanes in one block can land on the same node, so this
    stays an atomic read-modify-write per lane, as the framework's other mesh
    kernels do.
    """
    lines = []
    for index, name in enumerate(component):
        for node in range(n_nodes):
            lines.append("    for (int lane = 0; lane < ne; ++lane) {")
            lines.extend(
                _scatter_lines(
                    "out%s[bev%d[lane] * out_stride]" % (name, node),
                    "%sbout%d_%d[lane]" % (scale, index, node),
                    "      ",
                )
            )
            lines.append("    }")
    return lines


def _function_lines(name, template_params, signature, body):
    """A kernel that walks elements one at a time, on the IR spine.

    The compressed apply reads a half-precision store and converts as it goes,
    so it has no lane block to fill; it is still a `FunctionDefNode` with a real
    loop node, like everything else.
    """
    element = iterator("element", "ptrdiff_t")
    nodes = (
        FunctionDefNode(
            name="%s_impl" % name,
            params=tuple(line.strip().rstrip(",") for line in signature),
            body=(
                RawLinesNode(tuple(_parallel_loop_lines()), reason="element loop pragma"),
                LoopNode(
                    LoopKind.KERNEL,
                    element,
                    iteration_range(0, expr_ref("nelements", "mesh_extent")),
                    pre_increment(element),
                    body=(RawLinesNode(tuple(body), reason="kernel arithmetic"),),
                ),
                RawLinesNode(("", "  return SFEM_SUCCESS;"), reason="status"),
            ),
            return_type="int",
            qualifier="static SFEM_INLINE",
            template_params=tuple(template_params),
        ),
    )
    return list(render_kernel_ast_lines(name, nodes)) + [""]



#: The layout axis, as tables rather than branches.  A layout the plan does not
#: name is never looked up, which is what keeps emission a printer.
_LAYOUT_SUFFIX = {"standard": "", "packed_two_pass": "_packed_two_pass"}
_CONNECTIVITY_INDEX_TYPE = {"standard": "idx_t", "packed_two_pass": "uint16_t"}
_STORED_GATHERS_BY_LAYOUT = {
    "standard": lambda gathered, _component: _staged_gathers(gathered),
    "packed_two_pass": _packed_staged_gathers,
}
_STORED_SCATTER_BY_LAYOUT = {
    "standard": _blocked_scatter,
    "packed_two_pass": _packed_scatter,
}
_STORED_PROLOGUE_BY_LAYOUT = {
    "standard": _standard_stored_prologue,
    "packed_two_pass": _packed_signature,
}
_STORED_SKELETON_BY_LAYOUT = {
    "standard": lambda name, template, signature, scratch, gathers, compute, store, _c: (
        _blocked_function_lines(name, template, signature, scratch, gathers, compute, store)
    ),
    "packed_two_pass": _packed_function_lines,
}


#: The block width the published C entry points instantiate with.
_ABI_VECTOR_SIZE = 16

#: How the tangent is stored at the ABI boundary.  These are SFEM's own types
#: for exactly this object: `metric_tensor_t` is what the hand-written partial
#: assembly stores, and `compressed_t` with a `scaling_t` per element is what it
#: compresses to.  Emitting the pair `<name>` and `<name>_float` is what makes
#: the dispatch layer collapse them into one runtime-typed entry point.
_ABI_SCALARS = (("", "double"), ("_float", "float"))


def _abi_stream(scalar, role, component, const="const "):
    lines = ["    const ptrdiff_t %s_stride," % role]
    lines.extend(
        "    %s%s *const RSTR %s%s," % (const, scalar, role, name)
        for name in component
    )
    return lines


def _abi_geometry(dim):
    lines = [
        "    const geom_t *const RSTR g_adj%d," % index
        for index in range(dim * dim)
    ]
    lines.append("    const geom_t *const RSTR g_det0,")
    return lines


def _abi_tangent_arguments(dim, n_nodes, component):
    return ["adjugate%d" % i for i in range(dim * dim)]


def _packed_abi_prologue(scalar):
    """The packed ABI's leading parameters, in the order every packed kernel takes.

    Spelled here as well as in the templated kernel because the ABI is a C
    boundary: the dispatch layer builds its calls out of these names.
    """
    return [
        "    const ptrdiff_t n_packs,",
        "    const ptrdiff_t n_elements_per_pack,",
        "    const ptrdiff_t nelements,",
        "    const ptrdiff_t max_nodes_per_pack,",
        "    uint16_t **const RSTR elements,",
        "    const ptrdiff_t *const RSTR owned_nodes_ptr,",
        "    const ptrdiff_t n_ghost_entries,",
        "    const ptrdiff_t n_ghost_reduce_rows,",
        "    const ptrdiff_t *const RSTR ghost_ptr,",
        "    const idx_t *const RSTR ghost_idx,",
        "    const ptrdiff_t *const RSTR ghost_reduce_ptr,",
        "    const ptrdiff_t *const RSTR ghost_reduce_idx,",
        "    const idx_t *const RSTR ghost_reduce_dest,",
        "    %s *const RSTR ghost_buf," % scalar,
        "    const ptrdiff_t tangent_component_stride,",
        "    const metric_tensor_t *const RSTR tangent,",
    ]


def _c_abi_lines(
    prefix, dim, n_nodes, component, parameters, used_previous, reads_state,
    packed_layouts=(),
):
    """`extern "C"` wrappers, so the split is reachable from SFEM.

    The templated kernels above are what the generator produces; these are what
    the library links against.  Each is emitted twice, once per scalar type,
    because the dispatch layer collapses a `<name>`/`<name>_float` pair into a
    single public entry point that takes the scalar type as a
    `smesh::PrimitiveType` and the buffers as `void *` -- the shape the rest of
    SFEM's C ABI already has.
    """
    geometry_call = ", ".join(
        ["nelements"]
        + _CONNECTIVITY_CALL_BY_USE[bool(reads_state or used_previous)]
        + ["g_adj%d" % index for index in range(dim * dim)]
        + ["g_det0"]
    )
    previous_signature = _PREVIOUS_ABI_BY_USE[bool(used_previous)]
    previous_call = _PREVIOUS_CALL_BY_USE[bool(used_previous)]

    lines = []
    for suffix, scalar in _ABI_SCALARS:
        # --- the partial assembly ---------------------------------------
        name = "%s_inexact_apply_tangent_a_msoa%s" % (prefix, suffix)
        gathers = bool(reads_state or used_previous)
        lines.append('extern "C" int %s(' % name)
        lines.append("    const ptrdiff_t nelements,")
        lines.extend(_CONNECTIVITY_ABI_BY_USE[gathers])
        lines.extend(_abi_geometry(dim))
        lines.extend("    const %s %s," % (scalar, p) for p in parameters)
        lines.extend(_ABI_STATE_BY_USE[bool(reads_state)](scalar, component))
        lines.extend(previous_signature(scalar, component))
        lines.extend(
            [
                "    const ptrdiff_t tangent_component_stride,",
                "    metric_tensor_t *const RSTR tangent",
                ") {",
                "  return sfem::codegen::%s_inexact_apply_tangent_a_msoa_impl<"
                "%s, geom_t, metric_tensor_t, %d>("
                % (prefix, scalar, _ABI_VECTOR_SIZE),
                "      %s," % geometry_call,
                "      %s," % ", ".join(parameters),
                *_STATE_CALL_BY_USE[bool(reads_state)](component),
                "      %stangent_component_stride, tangent);"
                % previous_call(component),
                "}",
                "",
            ]
        )

        # --- the apply, from a `metric_tensor_t` store --------------------
        name = "%s_inexact_apply_stored_a_msoa%s" % (prefix, suffix)
        lines.append('extern "C" int %s(' % name)
        lines.extend(
            [
                "    const ptrdiff_t nelements,",
                "    idx_t **const RSTR elements,",
                "    const ptrdiff_t tangent_component_stride,",
                "    const metric_tensor_t *const RSTR tangent,",
            ]
        )
        lines.extend(_abi_stream(scalar, "h", component))
        lines.extend(_abi_stream(scalar, "out", component, const=""))
        lines[-1] = lines[-1].rstrip(",")
        lines.extend(
            [
                ") {",
                "  return sfem::codegen::%s_inexact_apply_stored_a_msoa_impl<"
                "%s, metric_tensor_t, %d>(" % (prefix, scalar, _ABI_VECTOR_SIZE),
                "      nelements, elements,",
                "      tangent_component_stride, tangent,",
                "      h_stride, %s," % ", ".join("h%s" % n for n in component),
                "      out_stride, %s);" % ", ".join("out%s" % n for n in component),
                "}",
                "",
            ]
        )

        # --- the apply, over a packed mesh --------------------------------
        for layout in packed_layouts:
            name = "%s_inexact_apply_stored%s_a_msoa%s" % (
                prefix, _LAYOUT_SUFFIX[layout], suffix
            )
            lines.append('extern "C" int %s(' % name)
            lines.extend(_packed_abi_prologue(scalar))
            lines.extend(_abi_stream(scalar, "h", component))
            lines.extend(_abi_stream(scalar, "out", component, const=""))
            lines[-1] = lines[-1].rstrip(",")
            lines.extend(
                [
                    ") {",
                    "  return sfem::codegen::%s_inexact_apply_stored%s_a_msoa_impl<"
                    "%s, metric_tensor_t, %d>("
                    % (prefix, _LAYOUT_SUFFIX[layout], scalar, _ABI_VECTOR_SIZE),
                    "      n_packs, n_elements_per_pack, nelements, max_nodes_per_pack,",
                    "      elements, owned_nodes_ptr, n_ghost_entries, n_ghost_reduce_rows,",
                    "      ghost_ptr, ghost_idx, ghost_reduce_ptr, ghost_reduce_idx,",
                    "      ghost_reduce_dest, ghost_buf,",
                    "      tangent_component_stride, tangent,",
                    "      h_stride, %s," % ", ".join("h%s" % n for n in component),
                    "      out_stride, %s);" % ", ".join("out%s" % n for n in component),
                    "}",
                    "",
                ]
            )

        # --- the apply, from a compressed store ---------------------------
        name = "%s_inexact_apply_compressed_a_msoa%s" % (prefix, suffix)
        lines.append('extern "C" int %s(' % name)
        lines.extend(
            [
                "    const ptrdiff_t nelements,",
                "    idx_t **const RSTR elements,",
                "    const ptrdiff_t tangent_component_stride,",
                "    const compressed_t *const RSTR tangent,",
                "    const scaling_t *const RSTR scaling,",
            ]
        )
        lines.extend(_abi_stream(scalar, "h", component))
        lines.extend(_abi_stream(scalar, "out", component, const=""))
        lines[-1] = lines[-1].rstrip(",")
        lines.extend(
            [
                ") {",
                "  return sfem::codegen::%s_inexact_apply_compressed_a_msoa_impl<"
                "%s, compressed_t, scaling_t>(" % (prefix, scalar),
                "      nelements, elements,",
                "      tangent_component_stride, tangent, scaling,",
                "      h_stride, %s," % ", ".join("h%s" % n for n in component),
                "      out_stride, %s);" % ", ".join("out%s" % n for n in component),
                "}",
                "",
            ]
        )
    return lines


#: The previous state reaches the ABI only where the material reads one.
_PREVIOUS_ABI_BY_USE = {
    True: lambda scalar, component: _abi_stream(scalar, "z", component),
    False: lambda scalar, component: [],
}
_PREVIOUS_CALL_BY_USE = {
    True: lambda component: "z_stride, %s, " % ", ".join("z%s" % n for n in component),
    False: lambda component: "",
}


def inexact_apply_files(material, unit, context):
    """The opt-in header for one unit, or ``()`` when the path does not apply.

    The header is named for the unit rather than the material, because a
    material with several units would otherwise have them collide on one path.
    """
    collection = getattr(unit, "form_collection", None)
    if collection is None:
        return ()
    dim = int(unit.dim)
    built = flux_form_for_collection(collection, dim)
    if built is None:
        return ()
    flux_form, is_deformation_gradient = built
    rule = context.specialization.quadrature_rule
    emitted = inexact_apply_kernel_source(
        _unit_name(material, unit),
        material.op_name,
        context.element_type,
        dim,
        int(rule.n_shape),
        flux_form,
        rule,
        flux_form.parameters,
        is_deformation_gradient,
    )
    if emitted is None:
        return ()
    _function, header, operator_source = emitted
    stem = "%s_%s_inexact_apply" % (
        _unit_name(material, unit),
        str(context.element_type).lower(),
    )
    return (
        ("%s_inline.hpp" % stem, header),
        ("%s_operator.cpp" % stem, operator_source),
    )


def _unit_name(material, unit):
    """What this unit's kernels are called, matching the other emitters."""
    return str(getattr(unit, "name", None) or material.name)
