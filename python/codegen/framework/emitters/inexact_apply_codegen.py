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
from codegen.framework.emitters.kernel_diagnostics_record import (
    DiagnosticsRecord,
    diagnostics_accessor_lines,
    diagnostics_record_lines,
)
from codegen.framework.emitters.kernel_prologue import kernel_constant
from codegen.framework.emitters.runtime_typed_abi import (
    cast_arguments,
    parameter_name,
    runtime_typed_entry_point_lines,
    runtime_typed_parameters,
)
from codegen.framework.emitters.quadrature_codegen import (
    REFERENCE_AXES,
    cpp_scalar_literal,
    quadrature_reference_accessor,
    reference_include_lines,
    tensor_product_q_index_lines,
    tensor_product_quadrature_weight_expr,
)
from codegen.framework.plans.conventions import inexact_apply_name
from codegen.framework.plans.kernel_signature import (
    PACKED_MESH_REDUCE_ONLY_ARGUMENTS,
)
from codegen.framework.plans.layout import cartesian_twin, gather_shape_order
from codegen.framework.plans.streams import (
    component_field_role,
    component_stream_names,
)
from codegen.framework.emitters.ast_printer import (
    CLikeKernelASTPrinter,
    lane_loop_header_lines,
    render_kernel_ast_lines,
)
from codegen.framework.ir.kernel_ast import (
    FunctionDefNode,
    LoopHeaderNode,
    LoopKind,
    LoopNode,
    RawLinesNode,
    add_assign_increment,
    expr_ref,
    iteration_range,
    iterator,
    pre_increment,
)
from codegen.framework.plans.flops import (
    expression_cost,
    inexact_apply_element_flops,
    inexact_tangent_element_flops,
)
from codegen.framework.plans.evaluation_strategy import (
    EvaluationStrategy,
    evaluation_strategy,
)
from codegen.framework.plans.inexact_apply import (
    action_stages,
    basis_gradient_symbols,
    emittable_inexact_apply_plan,
    flux_form_for_collection,
    geometry_argument_names,
    geometry_parameters,
    physical_field_gradient_definitions,
    projected_tangent,
    quadrature_accumulation,
    quadrature_measure,
    reference_field_gradient_definitions,
    reference_field_gradient_symbols,
    reference_gradients_by_point,
    tensor_product_weight_measure,
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


#: The component letters, in the order every stream and every symbol uses them.
_COMPONENT_NAMES = ("x", "y", "z")


def _adjugate_and_determinant(dim):
    """The geometry symbols the projected tangent is written in."""
    return (
        sp.Matrix(dim, dim, lambda r, c: sp.Symbol("adjugate%d" % (r * dim + c))),
        sp.Symbol("determinant"),
    )


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
    component = list(_COMPONENT_NAMES[:dim])
    adjugate, determinant = _adjugate_and_determinant(dim)
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

    integrand, gradient_symbols, previous_gradient_symbols = projected_tangent(
        plan,
        flux_form,
        rule,
        adjugate,
        determinant,
        is_deformation_gradient,
    )
    quadrature_weights = quadrature_accumulation(rule)
    reference = reference_gradients_by_point(rule, n_nodes, dim)
    # Which state values the tangent actually reads.  A state-independent
    # material -- linear elasticity, whose tangent is constant -- reads none,
    # and then the kernel must not gather a state it will not use.  The plan
    # answers this; emission spells the answer.
    used_state = plan.state_dependence(integrand, gradient_symbols)
    used_previous = plan.state_dependence(integrand, previous_gradient_symbols)

    tangent_symbols = [
        sp.Symbol("tangent%d" % slot) for slot in range(plan.tangent_components)
    ]
    stages = action_stages(plan, tangent_symbols, increment, output)
    action_body = []
    action_body_expressions = []
    # A stage's names are read by the stages after it and nowhere else, so a
    # name that CSE reduced to a bare temporary can be substituted into those
    # rather than declared.  `aliases` carries the mapping forward.
    aliases = {}
    for stage in stages:
        assignments = [
            (symbol, expression.xreplace(aliases))
            for symbol, expression in stage.assignments
        ]
        action_body.extend(
            _assignment_lines(assignments, stage.name, aliases=aliases)
        )
        action_body_expressions.extend(
            expression for _symbol, expression in assignments
        )

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
            used_previous, integrand, state, previous_state, gradient_symbols,
            previous_gradient_symbols, adjugate, determinant,
            quadrature_weights, reference, rule, plan,
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
        '#include "kernel_diagnostics.hpp"',
        "",
        '#include "%s_inexact_apply_inline.hpp"' % prefix,
        "",
    ]
    operator_lines.extend(
        _diagnostics_lines(
            prefix, plan, rule, dim, n_nodes, len(parameters), integrand,
            action_body_expressions, bool(used_state),
        )
    )
    operator_lines.extend(
        _c_abi_lines(
            prefix, dim, n_nodes, component, parameters, used_previous,
            bool(used_state),
            tuple(l for l in plan.apply_layouts if l != "standard"),
        )
    )
    return (
        inexact_apply_name(prefix, "tangent"),
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


def _diagnostics_lines(
    prefix, plan, rule, dim, n_nodes, n_parameters, integrand,
    action_expressions, reads_state,
):
    """One `KernelDiagnostics` record per published kernel.

    The goal asks for a FLOP and arithmetic-intensity function on every kernel
    "so performance reports are generated rather than written".  This family
    published none, so the roofline that motivated the variant had to be
    assembled by hand against numbers nobody could check.

    The record's shape and its field order are
    `emitters/kernel_diagnostics_record`; the costs are `plans/flops`.  Nothing
    is counted here.
    """
    n_qp = int(getattr(rule, "n_qp", 1) or 1)
    tangent_flops = inexact_tangent_element_flops(
        plan.element_type, dim, n_qp, n_nodes, dim, rule, integrand,
        plan.tangent_components, reads_state,
    )
    stored_flops = inexact_apply_element_flops(action_expressions)
    # The compressed apply scales the outputs rather than the forty-five stored
    # components: the action is linear in `Sbar`, so it is the same number
    # arrived at with fewer multiplies.
    compressed_flops = inexact_apply_element_flops(
        action_expressions, output_scale_flops=dim * n_nodes
    )

    lines = ["namespace sfem {", "namespace codegen {", ""]
    accessors = []
    for suffix, flops, points, u_streams, output_streams in (
        ("tangent", tangent_flops, n_qp, dim * n_nodes if reads_state else 0, 0),
        ("stored", stored_flops, 0, 0, dim * n_nodes),
        ("compressed", compressed_flops, 0, 0, dim * n_nodes),
    ):
        name = inexact_apply_name(prefix, suffix)
        lines.extend(
            diagnostics_record_lines(
                DiagnosticsRecord(
                    public_name=name,
                    element_type=str(rule.element_type),
                    dim=dim,
                    n_qp=points,
                    n_shape=n_nodes,
                    vector_size=_ABI_VECTOR_SIZE,
                    quadrature_order=int(getattr(rule, "order", 0) or 0),
                    cost=_DIAGNOSTIC_COST[suffix](integrand, action_expressions),
                    affine_mesh_flops_per_element=flops.affine_mesh_flops_per_element,
                    isoparametric_mesh_flops_per_element=(
                        flops.isoparametric_mesh_flops_per_element
                    ),
                    # The store is the kernel's own geometry: the tangent writes
                    # it and both applies read it instead of an adjugate.
                    geometry_streams=_DIAGNOSTIC_GEOMETRY[suffix](dim),
                    reference_scalars=0,
                    quadrature_weight_scalars=points,
                    material_scalars=_DIAGNOSTIC_MATERIAL[suffix](n_parameters),
                    u_streams=u_streams,
                    h_streams=output_streams,
                    output_streams=output_streams,
                    output_reads_per_element=output_streams,
                    output_writes_per_element=output_streams,
                )
            )
        )
        lines.append("")
        accessors.extend(diagnostics_accessor_lines(name))
    lines.extend(["} // namespace codegen", "} // namespace sfem", ""])
    lines.extend(accessors)
    lines.append("")
    return lines


#: What each kernel's per-operation counts are taken from.  The tangent's are
#: the integrand it evaluates at every point; both applies' are the contraction
#: they perform once.
#: The tangent evaluates its integrand at every point, so its counts are
#: per-point and the record's `n_qp` multiplies them.  An apply has no points:
#: its counts are the element's, and they are reported beside an `n_qp` of zero
#: so that `n_qp * flops_per_qp + mesh_flops` still totals the kernel.
_DIAGNOSTIC_COST = {
    "tangent": lambda integrand, action: expression_cost(integrand),
    "stored": lambda integrand, action: expression_cost(action),
    "compressed": lambda integrand, action: expression_cost(action),
}

#: The material constants each kernel is handed.  The applies take none: the
#: material has been evaluated away into the store, which is the whole point.
_DIAGNOSTIC_MATERIAL = {
    "tangent": lambda n_parameters: n_parameters,
    "stored": lambda n_parameters: 0,
    "compressed": lambda n_parameters: 0,
}

#: How many element-indexed streams each kernel is handed.  The tangent takes
#: the adjugate and the determinant; the applies take the stored tangent
#: instead, which is what makes them cheap to feed.
_DIAGNOSTIC_GEOMETRY = {
    "tangent": lambda dim: dim * dim + 1,
    "stored": lambda dim: 0,
    "compressed": lambda dim: 0,
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


def _assignment_lines(assignments, prefix, indent="    ", aliases=None):
    """Common subexpressions first, then the named values, as C declarations.

    When `aliases` is a dict, a named value whose reduced expression is a bare
    symbol is not declared at all -- the mapping is recorded there instead, for
    the caller to substitute into whatever reads it.  CSE hoists a shared
    subexpression into a temporary and leaves the name that asked for it holding
    nothing but that temporary, so the declaration is `const s_t pa_g0_0_0_0 =
    reference_product_t10;` and the only thing it adds is a second name for one
    value.  There were 540 of those in each of the two PROTEUS_HEX8
    inexact-apply headers.

    It is opt-in because dropping a declaration is only safe where the caller
    controls every use of it.  The stage loop does -- a stage's names are read
    by the stages after it and nowhere else -- and the three other callers here
    do not, so they pass nothing and keep their aliases.

    Only a bare symbol is eliminated.  A negation such as `-reference_product_t10`
    is left as its own declaration: substituting it would push the negation into
    each of the three output terms that read it, which is more arithmetic
    spelled, not less.
    """
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
    for symbol, expression in zip(symbols, reduced):
        if aliases is not None and expression.is_Symbol:
            aliases[symbol] = expression
            continue
        lines.append(
            "%sconst s_t %s = %s;" % (indent, symbol, _sfem_ccode(expression))
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


def _geometry_arguments(dim):
    """The geometry the tangent takes, named by `plans/inexact_apply`."""
    return [
        "    %s," % parameter
        for parameter in geometry_parameters(dim, "const g_t *const RSTR")
    ]


#: The boundary names of the four field roles these kernels carry.  The
#: prefixes are `plans/streams.COMPONENT_FIELD_STREAMS`, which is where the
#: convention is stated -- one vector field's components as separate streams,
#: role first and component last.  The emitter reads the prefixes off the plan
#: rather than repeating the letters, so a role renamed there reaches the
#: signature, the gather and the C boundary together.
_FIELD_ROLE = {
    name: component_field_role(name)
    for name in ("current", "previous", "direction", "output")
}
_CURRENT = _FIELD_ROLE["current"].prefix
_PREVIOUS = _FIELD_ROLE["previous"].prefix
_DIRECTION = _FIELD_ROLE["direction"].prefix
_OUTPUT = _FIELD_ROLE["output"].prefix

#: Which role each staging prefix belongs to, so a site holding the prefix can
#: still ask the plan for the names.
_ROLE_BY_PREFIX = {role.prefix: role for role in _FIELD_ROLE.values()}


def _stream_declarations(prefix, component, declaration="const s_t *const RSTR"):
    """One role at a kernel boundary: its stride, then one buffer per component.

    The names come from `plans/streams`; only how C spells a pointer is this
    emitter's business.
    """
    stride, *buffers = component_stream_names(_ROLE_BY_PREFIX[prefix], component)
    return ["const ptrdiff_t %s" % stride] + [
        "%s %s" % (declaration, name) for name in buffers
    ]


def _stream_arguments(role, component):
    return ["    %s," % line for line in _stream_declarations(role, component)]


def _output_arguments(component):
    lines = _stream_declarations(_OUTPUT, component, "s_t *const RSTR")
    return ["    %s," % line for line in lines[:-1]] + ["    %s" % lines[-1]]


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
    references = [_ReferenceTable("grad_ref_%s" % axis) for axis in REFERENCE_AXES[:dim]]
    references.append(_ReferenceTable("q_weight"))
    return list(reference_include_lines(rule, references))


def _tensor_product_reference_includes(rule, dim):
    """The one-dimensional tables, and the contraction that reads them.

    A tensor-product element's shared reference data is one-dimensional --
    `ref_line_p1_q2<s_t>::shape_1d()` and `grad_1d()`, and the rule's
    `q_weight_1d()` beside them -- because sum factorization never needs the
    element's full reference gradients.  `tensor_product_kernels.hpp` carries
    the contraction itself, and it is the same one the energy path calls.
    """
    references = [_ReferenceTable("shape_1d"), _ReferenceTable("q_weight_1d")]
    return list(reference_include_lines(rule, references)) + [
        '#include "tensor_product_kernels.hpp"'
    ]


def _no_reference_includes(rule, dim):
    """A kernel that reads no shared table includes none."""
    return []


#: What each strategy forwards into.  An EXPANDED element folds its reference
#: gradients into the arithmetic and reads nothing at run time.
_REFERENCE_INCLUDES_BY_STRATEGY = {
    EvaluationStrategy.QUADRATURE: _shared_reference_includes,
    EvaluationStrategy.EXPANDED: _no_reference_includes,
    EvaluationStrategy.SUM_FACTORIZED: _tensor_product_reference_includes,
}


@dataclasses.dataclass(frozen=True)
class _GradientRole:
    """A field whose gradient the projected tangent reads.

    Two at most: the current state, and -- for a rate-dependent material such as
    Kelvin-Voigt viscosity -- the previous one.  They differ in their name and in
    nothing else, which is why every step below loops over them rather than
    spelling the current state and then spelling it again.
    """

    name: str
    nodal: tuple
    reference: tuple
    gradient: tuple


def _gradient_roles(dim, state, previous_state, gradient_symbols,
                    previous_gradient_symbols, used_state, used_previous):
    """The fields this tangent actually reads, in signature order.

    `plan.state_dependence` answered which; this pairs each with the symbols its
    two gradient stages are named by.
    """
    roles = []
    for name, nodal, gradient, used in (
        ("u", state, gradient_symbols, used_state),
        ("z", previous_state, previous_gradient_symbols, used_previous),
    ):
        if not used:
            continue
        roles.append(
            _GradientRole(
                name=name,
                nodal=tuple(tuple(row) for row in nodal),
                reference=reference_field_gradient_symbols(dim, "g%s" % name),
                gradient=gradient,
            )
        )
    return tuple(roles)


@dataclasses.dataclass(frozen=True)
class _GradientSource:
    """Where one point's physical field gradients come from.

    The three evaluation strategies differ in this and in nothing else: the
    material arithmetic below is the same expression in `gu_<c>_<a>` whatever
    produced them.  Keeping the difference in one object is what lets the body
    be written once.
    """

    tables: tuple = ()
    scratch: tuple = ()
    gathers: tuple = ()
    q_prologue: tuple = ()
    lane_lines: tuple = ()
    #: The gradient stages, in order, each a list of (symbol, expression) pairs.
    #: Separate blocks and not one list: the second stage reads what the first
    #: defines, and a common-subexpression pass over both together would hoist a
    #: temporary above the definition it depends on.
    stages: tuple = ()
    weight: object = None


def _physical_definitions(roles, adjugate, determinant, dim):
    """The second gradient stage, which every strategy shares."""
    definitions = []
    for role in roles:
        definitions.extend(
            physical_field_gradient_definitions(
                role.reference, role.gradient, adjugate, determinant, dim
            )
        )
    return tuple(definitions)


def _explicit_reference_definitions(roles, basis_symbols, n_nodes, dim):
    """The first gradient stage, for an element that evaluates its basis."""
    definitions = []
    for role in roles:
        definitions.extend(
            reference_field_gradient_definitions(
                role.nodal, basis_symbols, role.reference, n_nodes, dim
            )
        )
    return tuple(definitions)


def _nodal_staging(roles, component, n_nodes):
    """Every nodal value of every field the tangent reads.

    All of them, rather than the ones a free-symbol scan finds: the reference
    gradient of a component is a contraction over that component's nodes, so a
    component the material reads is read at every node of the element.
    """
    gathered = []
    for role in roles:
        gathered.extend(
            _gathered_names(
                role.name, component, n_nodes, _all_names(role.name, component, n_nodes)
            )
        )
    return gathered


def _explicit_gradient_source(
    roles, component, basis_reads, tables, weight, n_nodes, dim,
):
    """A source that stages the nodal values and contracts them in the kernel.

    Shared by the two strategies that evaluate the basis explicitly: an
    EXPANDED element, whose basis gradients are literals, and a QUADRATURE
    element, which reads them from the shared reference struct inside the loop.
    """
    gathered = _nodal_staging(roles, component, n_nodes)
    scratch = _CONNECTIVITY_BY_USE[bool(gathered)](n_nodes)
    scratch.extend("    s_t b%s[VS];" % value for value, _s, _n, _r in gathered)
    lane_lines = [
        "      const s_t %s = b%s[lane];" % (value, value)
        for value, _s, _n, _r in gathered
    ]
    lane_lines.extend(basis_reads)
    return _GradientSource(
        tables=tuple(tables),
        scratch=tuple(scratch),
        gathers=tuple(_STAGED_GATHERS_BY_USE[bool(gathered)](gathered)),
        lane_lines=tuple(lane_lines),
        weight=weight,
    )


def _expanded_gradient_source(
    roles, component, reference, weights, n_nodes, dim, rule, plan,
):
    """An EXPANDED element: one point, and the basis gradients are numbers.

    `plans.evaluation_strategy` says a lowest-order simplex evaluates in closed
    form -- "no quadrature loop, no per-point geometry, no reference-basis
    tables" -- so the first gradient stage is the nodal contraction with those
    numbers substituted, and the point's weight is a rational the tangent
    carries rather than a table lookup.
    """
    basis_symbols = basis_gradient_symbols(n_nodes, dim)
    literals = {
        basis_symbols[node][axis]: reference[0][node][axis]
        for node in range(n_nodes)
        for axis in range(dim)
    }
    definitions = tuple(
        (symbol, expression.xreplace(literals))
        for symbol, expression in _explicit_reference_definitions(
            roles, basis_symbols, n_nodes, dim
        )
    )
    source = _explicit_gradient_source(
        roles, component, (), (), weights[0], n_nodes, dim
    )
    return dataclasses.replace(source, stages=(definitions,))


def _tabulated_gradient_source(
    roles, component, reference, weights, n_nodes, dim, rule, plan,
):
    """A QUADRATURE element: the basis gradients come from the shared struct.

    `ref_<element>_<rule><s_t>::grad_ref_x()` and `quad_<cell>_<rule>::q_weight()`
    are what every other kernel on this element reads, and reading them here is
    what `quadrature_reference_accessor` exists for.
    """
    basis_symbols = basis_gradient_symbols(n_nodes, dim)
    tables = [kernel_constant("NQ", len(weights), indent="    ")]
    tables.extend(
        "    const s_t *const RSTR qgrad_%s = %s;"
        % (axis, quadrature_reference_accessor(rule, "grad_ref_%s" % axis))
        for axis in REFERENCE_AXES[:dim]
    )
    tables.append(
        "    const s_t *const RSTR qweight = %s;"
        % quadrature_reference_accessor(rule, "q_weight")
    )
    tables.extend(_measure_lines(quadrature_measure(rule)))
    basis_reads = _BASIS_READS_BY_USE[bool(roles)](basis_symbols, n_nodes, dim)
    source = _explicit_gradient_source(
        roles, component, basis_reads, tables, "qweight[q] * QMEASURE",
        n_nodes, dim,
    )
    return dataclasses.replace(
        source,
        stages=(
            _explicit_reference_definitions(roles, basis_symbols, n_nodes, dim),
        ),
    )


def _sum_factorized_gradient_source(
    roles, component, reference, weights, n_nodes, dim, rule, plan,
):
    """A tensor-product element: `tensor_gradient_contiguous` does the first stage.

    The strategy is sum factorization, so this kernel does what every other
    tensor-product kernel in the framework does -- stage the nodal values in the
    Cartesian shape order the contraction is written against, hand them to the
    shared contraction with the one-dimensional tables, and read the reference
    field gradient back out per point.  The element's full reference gradients
    are never formed, which is the whole point of the strategy and the reason
    this kernel no longer carries a private table of them.
    """
    n_qp = len(weights)
    # `plans/layout` owns which order a kernel's field streams are gathered in,
    # and its answer is identity for a PROTEUS element -- whose mesh already
    # numbers its nodes lexicographically.  Spelling the permutation here
    # instead would be right for HEX8 and wrong for its Cartesian twin.
    shape_order = gather_shape_order(plan.element_type, dim, n_nodes, True)
    tables = [
        kernel_constant("NQ", n_qp, indent="    "),
        kernel_constant("NQ1", rule.tensor_product_n_qp_1d, indent="    "),
    ]
    # The one-dimensional basis tables belong to the contraction, so a material
    # whose tangent reads no field -- linear elasticity's -- names none of them
    # and keeps the weights, which the quadrature sum needs whatever the
    # material is.
    tables.extend(_CONTRACTION_TABLES_BY_USE[bool(roles)](rule, n_nodes))
    tables.append(
        "    const s_t *const RSTR q_weight_1d = %s;"
        % quadrature_reference_accessor(rule, "q_weight_1d")
    )
    tables.extend(_measure_lines(tensor_product_weight_measure(rule, dim)))

    scratch = _CONNECTIVITY_BY_USE[bool(roles)](n_nodes)
    gathers = []
    q_prologue = list(tensor_product_q_index_lines(dim, "      "))
    lane_lines = []
    for role in roles:
        scratch.append("    s_t b%s_data[NS * %d][VS];" % (role.name, dim))
        scratch.append("    s_t g%s_ref_q[NQ * %d * VS];" % (role.name, dim * dim))
        gathers.extend(
            _lane_loop(
                [
                    "      b%s_data[%d][lane] = %s%s[bev%d[lane] * %s_stride];"
                    % (
                        role.name,
                        shape * dim + field,
                        role.name,
                        component[field],
                        shape_order[shape],
                        role.name,
                    )
                    for shape in range(n_nodes)
                    for field in range(dim)
                ]
            )
        )
        gathers.extend(
            "    tensor_gradient_contiguous<s_t, NQ, NS, VS, %d, %d>"
            "(ne, shape_1d, grad_1d, b%s_data, %d, &g%s_ref_q[%d * VS]);"
            % (dim, dim, role.name, field, role.name, field * n_qp * dim)
            for field in range(dim)
        )
        q_prologue.extend(
            "      const s_t *const RSTR g%s_ref%d = &g%s_ref_q[(%d + q * %d) * VS];"
            % (role.name, field * dim + axis, role.name, field * n_qp * dim + axis, dim)
            for field in range(dim)
            for axis in range(dim)
        )
        lane_lines.extend(
            "      const s_t %s = g%s_ref%d[lane];"
            % (role.reference[field][axis], role.name, field * dim + axis)
            for field in range(dim)
            for axis in range(dim)
        )
    return _GradientSource(
        tables=tuple(tables),
        scratch=tuple(scratch),
        gathers=tuple(gathers),
        q_prologue=tuple(q_prologue),
        lane_lines=tuple(lane_lines),
        weight="%s * QMEASURE" % tensor_product_quadrature_weight_expr(dim),
    )


def _basis_reads(basis_symbols, n_nodes, dim):
    """One point's reference basis gradients, from the shared struct."""
    return [
        "      const s_t %s = qgrad_%s[q * %d + %d];"
        % (basis_symbols[node][axis], REFERENCE_AXES[axis], n_nodes, node)
        for node in range(n_nodes)
        for axis in range(dim)
    ]


def _contraction_tables(rule, n_nodes):
    """What `tensor_gradient_contiguous` is given: the node count and the 1D bases."""
    return [
        kernel_constant("NS", n_nodes, indent="    "),
        "    const s_t *const RSTR shape_1d = %s;"
        % quadrature_reference_accessor(rule, "shape_1d"),
        "    const s_t *const RSTR grad_1d = %s;"
        % quadrature_reference_accessor(rule, "grad_1d"),
    ]


#: A tangent that reads no field reads no basis either.  Tables rather than
#: branches, as elsewhere here.
_BASIS_READS_BY_USE = {
    True: _basis_reads,
    False: lambda basis_symbols, n_nodes, dim: [],
}
_CONTRACTION_TABLES_BY_USE = {
    True: _contraction_tables,
    False: lambda rule, n_nodes: [],
}


def _gradient_stage_lines(source, indent):
    """The gradient stages, each with its own common-subexpression pass.

    One pass per stage rather than one over all of them: the physical gradient
    is a function of the reference gradient, so they are consecutive
    computations rather than independent outputs, and a single pass would name a
    temporary before the value it reads exists.
    """
    lines = []
    for index, stage in enumerate(source.stages):
        lines.extend(_assignment_lines(list(stage), "gradient%d" % index, indent=indent))
    return lines


def _measure_lines(measure):
    """The reciprocal total weight, as the one constant the point scales by."""
    return [
        "    static constexpr s_t QMEASURE = %s;"
        % cpp_scalar_literal(float(sp.sympify(measure).evalf(20)))
    ]


#: Where one point's physical field gradients come from, per evaluation
#: strategy.  A table, because the choice belongs to `plans.evaluation_strategy`
#: and emission only looks the answer up.
_GRADIENT_SOURCE_BY_STRATEGY = {
    EvaluationStrategy.EXPANDED: _expanded_gradient_source,
    EvaluationStrategy.QUADRATURE: _tabulated_gradient_source,
    EvaluationStrategy.SUM_FACTORIZED: _sum_factorized_gradient_source,
}

#: Whether a strategy's tangent needs a quadrature loop around its lane loop.
_NEEDS_QUADRATURE_LOOP = {
    EvaluationStrategy.EXPANDED: False,
    EvaluationStrategy.QUADRATURE: True,
    EvaluationStrategy.SUM_FACTORIZED: True,
}


def _expanded_tangent_body(
    integrand, integrand_symbols, source, lane_prologue, plan,
):
    """An EXPANDED element: one lane loop, nothing around it.

    The point's weight is a rational, so it multiplies the integrand here and
    the whole element is one straight-line block: gradients, material, store.
    """
    tangent_symbols = [
        sp.Symbol("tangent%d" % slot) for slot in range(plan.tangent_components)
    ]
    compute = list(lane_prologue)
    compute.extend(
        "  %s" % line for line in _gradient_stage_lines(source, "    ")
    )
    compute.extend(
        "  %s" % line
        for line in _assignment_lines(
            [
                (symbol, source.weight * expression)
                for symbol, expression in zip(tangent_symbols, integrand)
            ],
            "tangent",
        )
    )
    compute.extend(
        "      %s = tangent_t(tangent%d);" % (_BLOCKED_TANGENT_LANE % slot, slot)
        for slot in range(plan.tangent_components)
    )
    return compute, None


def _quadrature_point_lines(integrand, integrand_symbols, source, lane_prologue):
    """One quadrature point's contribution, for one lane.

    Straight-line: this is the body of a lane loop and carries no loop of its
    own.  The gradient stages come first because the material is a function of
    them and of nothing else, and they are CSEd together with it.
    """
    lines = list(lane_prologue)
    lines.append("        const s_t qw = %s;" % source.weight)
    # Two blocks, in this order and not one: the material is a function of the
    # gradients, so they are values it reads rather than results beside it, and
    # a single common-subexpression pass over both would hoist a temporary above
    # the definition it depends on.
    lines.extend("    %s" % line for line in _gradient_stage_lines(source, "        "))
    lines.extend(
        "    %s" % line
        for line in _assignment_lines(
            list(zip(integrand_symbols, integrand)), "integrand", indent="        "
        )
    )
    lines.extend(
        "        btangent_acc[%d][lane] += qw * integrand%d;" % (slot, slot)
        for slot in range(len(integrand_symbols))
    )
    return lines


def _quadrature_tangent_body(
    integrand, integrand_symbols, source, lane_prologue, plan,
):
    """A quadrature loop *around* three lane loops: zero, accumulate, write out.

    This is the framework's nesting -- lane loops innermost, holding only
    straight-line work -- and it is why the accumulator is a lane-major array
    rather than a scalar: it has to survive across the points while staying
    per-lane.  Accumulating in `s_t` and converting once at the end keeps the
    store's precision out of the sum.
    """
    target = current_target()
    pragma = target.vectorize_pragma() if hasattr(target, "vectorize_pragma") else None
    slots = plan.tangent_components
    zero = _lane_loop_node(
        ["        btangent_acc[%d][lane] = s_t(0);" % slot for slot in range(slots)],
        pragma,
    )
    point = _lane_loop_node(
        _quadrature_point_lines(
            integrand, integrand_symbols, source, lane_prologue
        ),
        pragma,
    )
    q = iterator("q", "int")
    accumulate = LoopNode(
        LoopKind.QUADRATURE,
        q,
        iteration_range(0, expr_ref("NQ", "quadrature_points")),
        pre_increment(q),
        body=(
            RawLinesNode(source.q_prologue, reason="point-invariant addresses"),
            point,
        ),
    )
    write_out = _lane_loop_node(
        [
            "        %s = tangent_t(btangent_acc[%d][lane]);"
            % (_BLOCKED_TANGENT_LANE % slot, slot)
            for slot in range(slots)
        ],
        pragma,
    )
    return [], [zero, accumulate, write_out]


#: Which body shape the tangent takes, keyed on whether its strategy needs a
#: quadrature loop.
_ACCUMULATOR_BY_LOOP = {
    False: lambda slots: [],
    True: lambda slots: ["    s_t btangent_acc[%d][VS];" % slots],
}

_TANGENT_BODY_BY_LOOP = {
    False: _expanded_tangent_body,
    True: _quadrature_tangent_body,
}


def _tangent_lines(
    prefix, dim, n_nodes, component, parameters, used_state, used_previous,
    integrand, state, previous_state, gradient_symbols, previous_gradient_symbols,
    adjugate, determinant, weights, reference, rule, plan,
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

    How the gradients reach that loop is `plans.evaluation_strategy`'s answer
    and nothing this file re-derives: a lowest-order simplex folds the basis
    gradients in as literals, a higher-order simplex reads the shared reference
    struct, and a tensor-product element contracts through `tensor_gradient`.

    Takes the previous state only when the material reads one, so a
    rate-independent material keeps the shorter signature.
    """
    integrand_symbols = [
        sp.Symbol("integrand%d" % slot) for slot in range(plan.tangent_components)
    ]
    strategy = evaluation_strategy(plan.element_type)
    roles = _gradient_roles(
        dim, state, previous_state, gradient_symbols, previous_gradient_symbols,
        used_state, used_previous,
    )
    source = _GRADIENT_SOURCE_BY_STRATEGY[strategy](
        roles, component, reference, weights, n_nodes, dim, rule, plan,
    )
    source = dataclasses.replace(
        source,
        stages=tuple(source.stages)
        + (_physical_definitions(roles, adjugate, determinant, dim),),
    )

    gathers = list(source.gathers)
    gathers.extend(_blocked_geometry_bases(dim))
    gathers.extend(_blocked_tangent_bases(plan.tangent_components))

    lane_prologue = _blocked_geometry_lines(dim) + list(source.lane_lines)
    compute, inner = _TANGENT_BODY_BY_LOOP[_NEEDS_QUADRATURE_LOOP[strategy]](
        integrand, integrand_symbols, source, lane_prologue, plan,
    )

    signature = ["    const ptrdiff_t nelements,"]
    signature.extend(_CONNECTIVITY_ARGUMENT_BY_USE[bool(roles)])
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
    tables = list(source.tables)
    # The accumulator survives the quadrature loop and stays per-lane, so it is
    # lane-major scratch.  Declared only where there is a loop to survive.
    tables.extend(
        _ACCUMULATOR_BY_LOOP[_NEEDS_QUADRATURE_LOOP[strategy]](plan.tangent_components)
    )
    return _blocked_function_lines(
        inexact_apply_name(prefix, "tangent"),
        ("typename s_t", "typename g_t", "typename tangent_t", "int VS"),
        signature, tables + list(source.scratch), gathers, compute, [], inner=inner,
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
        inexact_apply_name(prefix, "stored", _LAYOUT_TRAVERSAL[layout]),
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
        inexact_apply_name(prefix, "compressed"),
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
_CONNECTIVITY_ABI_BY_USE = {
    True: ["idx_t **const RSTR elements"],
    False: [],
}
_ABI_STATE_BY_USE = {
    True: lambda component: _abi_stream("u", component),
    False: lambda component: [],
}


def _all_names(role, component, n_nodes):
    return frozenset(
        sp.Symbol("%s%s_%d" % (role, name, node))
        for name in component
        for node in range(n_nodes)
    )


def _vectorize_pragma():
    """The bound target's lane-loop pragma, or `None` where it does not vectorise."""
    target = current_target()
    if target is None or not hasattr(target, "vectorize_pragma"):
        return None
    return target.vectorize_pragma()


def _lane_loop(body, indent="    "):
    """The lane loop, from the IR rather than from this emitter's own text.

    `LoopHeaderNode` exists for exactly this: a site whose body is still lines
    can take its header -- the vectorize pragma and the `for` -- from the same
    `LoopNode` a fully migrated site would print, and close the brace itself.
    `_work_item_loop_lines` in `emitters/energy_codegen.py` already does it this
    way, so this stops being a fourth spelling of one loop and becomes a second
    caller of the one the printer owns.
    """
    header = lane_loop_header_lines(_vectorize_pragma(), indent)
    return list(header) + list(body) + ["%s}" % indent]


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
    """The block's geometry as base pointers, one per stream.

    The geometry is element-indexed and therefore already contiguous across the
    block, so it needs no staging -- only a pointer to the block's first
    element, so the lane loop reads `bg_adj0[lane]` rather than rebuilding the
    address per lane.

    Which streams there are is `plans/inexact_apply.geometry_streams`, which
    forwards to `plans/geometry_quantities`; only the `b` prefix is this
    emitter's own.
    """
    return [
        "    const g_t *const RSTR b%s = %s + evb;" % (name, name)
        for name in geometry_argument_names(dim)
    ]


def _blocked_geometry_lines(dim, indent="      "):
    """The lane's geometry, by the local name the arithmetic is written in."""
    return [
        "%sconst s_t %s = s_t(b%s[lane]);" % (indent, local, name)
        for local, name in zip(
            ["adjugate%d" % index for index in range(dim * dim)] + ["determinant"],
            geometry_argument_names(dim),
        )
    ]


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

    The traversal arguments are
    `plans.kernel_signature.PACKED_MESH_REDUCE_ONLY_ARGUMENTS`, which is where
    the reasons live for the two the other emitters' packed kernels carry and
    these do not.  The stored tangent after them is this family's own.

    `component` and `n_components` are unread; they are here because
    `_STORED_PROLOGUE_BY_LAYOUT` calls both prologues the same way.
    """
    return [
        "    %s," % argument.declaration
        for argument in PACKED_MESH_REDUCE_ONLY_ARGUMENTS
    ] + [
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
    lines.extend(_lane_loop(["    %s" % line for line in compute], "        "))
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


def _lane_loop_node(body_lines, pragma):
    """A lane loop over the tile, with nothing nested inside it.

    Lane loops in this framework are innermost and hold straight-line work: a
    loop inside one is what `test_openmp_hot_loop_families_report_vectorized_local_loops`
    exists to catch, and a quadrature loop nested in a lane loop is the shape it
    caught here.
    """
    lane = iterator("lane", "int")
    return LoopNode(
        LoopKind.SIMD,
        lane,
        iteration_range(0, expr_ref("ne", "tile_extent")),
        pre_increment(lane),
        body=(RawLinesNode(tuple(body_lines), reason="kernel arithmetic"),),
        vectorized=bool(pragma),
    )


def _blocked_function_lines(name, template_params, signature, scratch, gathers, compute, store,
                            inner=None):
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
        *(inner if inner is not None else [_lane_loop_node(compute, pragma)]),
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
#: Which traversal each layout puts in the qualifier slot, as
#: `plans/conventions` spells it.  This was `_LAYOUT_SUFFIX`, holding
#: `"_packed_two_pass"` with the joining underscore baked in, which is what a
#: name built by string concatenation needs and what a name built by
#: `conventions.inexact_apply_name` must not be given.
_LAYOUT_TRAVERSAL = {"standard": "", "packed_two_pass": "packed_two_pass"}
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
#: compresses to.  None of them is the dispatched scalar -- the store's
#: precision is its own template axis, which is why `carries_scalar` leaves
#: them alone and why they cross the boundary spelled as themselves.
_ABI_TANGENT_STORE = "metric_tensor_t"


def _abi_stream(role, component, const="const "):
    """The same role at the C boundary, where the scalar is not a template."""
    return _stream_declarations(role, component, "%ss_t *const RSTR" % const)


def _abi_geometry(dim):
    """The affine geometry the tangent reads, as the C boundary declares it."""
    return list(geometry_parameters(dim))


def _packed_abi_prologue():
    """The packed ABI's leading parameters, in the order every packed kernel takes.

    Spelled here as well as in the templated kernel because the ABI is a C
    boundary: the dispatch layer builds its calls out of these names.
    """
    return [
        argument.declaration for argument in PACKED_MESH_REDUCE_ONLY_ARGUMENTS
    ] + [
        "const ptrdiff_t tangent_component_stride",
        "const %s *const RSTR tangent" % _ABI_TANGENT_STORE,
    ]


def _abi_entry_point(name, params, template, template_arguments):
    """One `extern "C"` entry point over the template, typed at run time.

    Stands where `for suffix, scalar in _ABI_SCALARS:` stood.  A body written in
    terms of `s_t` is the same text for every precision and is emitted once as a
    template above; publishing a symbol per precision on top of it restated the
    whole parameter list a second time and made the dispatch declare and switch
    over both.  The entry point takes the scalar's width and selects the
    instantiation itself, which is the shape the rest of SFEM's C ABI has.

    Every kernel here passes its arguments in parameter order, but the cast is
    applied by name through `cast_arguments` rather than by position, so a call
    list that drifted from its signature would fail to compile instead of
    quietly mis-casting a buffer.
    """
    arguments = [parameter_name(param) for param in params]
    return runtime_typed_entry_point_lines(
        name,
        params,
        lambda scalar_type, _positional: [
            "return sfem::codegen::%s<%s>(" % (template, template_arguments(scalar_type)),
            "    %s);" % ", ".join(cast_arguments(params, arguments, scalar_type)),
        ],
    )


@dataclasses.dataclass(frozen=True)
class _AbiEntryPoint:
    """One published symbol: its name, its parameters, and what it reaches."""

    name: str
    params: tuple
    template: str
    template_arguments: object
    element_type: str = "idx_t"


def _c_abi_entry_points(
    prefix, dim, n_nodes, component, parameters, used_previous, reads_state,
    packed_layouts=(), template_prefix=None,
):
    """The four symbols this element publishes, in the order it publishes them.

    One list, two consumers: the element that owns the micro-kernel turns it
    into definitions, and the one that delegates turns it into prototypes for
    the alias pass to define.  Spelling the parameter lists twice is how the
    two would come to disagree, and the dispatch builds its calls out of them.
    """
    template_prefix = prefix if template_prefix is None else template_prefix
    gathers = bool(reads_state or used_previous)
    entry_points = []

    # --- the partial assembly -------------------------------------------
    params = ["const ptrdiff_t nelements"]
    params.extend(_CONNECTIVITY_ABI_BY_USE[gathers])
    params.extend(_abi_geometry(dim))
    params.extend("const s_t %s" % name for name in parameters)
    params.extend(_ABI_STATE_BY_USE[bool(reads_state)](component))
    params.extend(_PREVIOUS_ABI_BY_USE[bool(used_previous)](component))
    params.extend(
        [
            "const ptrdiff_t tangent_component_stride",
            "%s *const RSTR tangent" % _ABI_TANGENT_STORE,
        ]
    )
    entry_points.append(
        _AbiEntryPoint(
            name=inexact_apply_name(prefix, "tangent"),
            params=tuple(params),
            template="%s_impl" % inexact_apply_name(template_prefix, "tangent"),
            template_arguments=lambda scalar_type: "%s, geom_t, %s, %d"
            % (scalar_type, _ABI_TANGENT_STORE, _ABI_VECTOR_SIZE),
        )
    )

    # --- the apply, from a `metric_tensor_t` store ------------------------
    params = [
        "const ptrdiff_t nelements",
        "idx_t **const RSTR elements",
        "const ptrdiff_t tangent_component_stride",
        "const %s *const RSTR tangent" % _ABI_TANGENT_STORE,
    ]
    params.extend(_abi_stream("h", component))
    params.extend(_abi_stream("out", component, const=""))
    entry_points.append(
        _AbiEntryPoint(
            name=inexact_apply_name(prefix, "stored"),
            params=tuple(params),
            template="%s_impl" % inexact_apply_name(template_prefix, "stored"),
            template_arguments=lambda scalar_type: "%s, %s, %d"
            % (scalar_type, _ABI_TANGENT_STORE, _ABI_VECTOR_SIZE),
        )
    )

    # --- the apply, over a packed mesh ------------------------------------
    for layout in packed_layouts:
        params = list(_packed_abi_prologue())
        params.extend(_abi_stream("h", component))
        params.extend(_abi_stream("out", component, const=""))
        traversal = _LAYOUT_TRAVERSAL[layout]
        entry_points.append(
            _AbiEntryPoint(
                name=inexact_apply_name(prefix, "stored", traversal),
                params=tuple(params),
                template="%s_impl"
                % inexact_apply_name(template_prefix, "stored", traversal),
                template_arguments=lambda scalar_type: "%s, %s, %d"
                % (scalar_type, _ABI_TANGENT_STORE, _ABI_VECTOR_SIZE),
                element_type="uint16_t",
            )
        )

    # --- the apply, from a compressed store -------------------------------
    params = [
        "const ptrdiff_t nelements",
        "idx_t **const RSTR elements",
        "const ptrdiff_t tangent_component_stride",
        "const compressed_t *const RSTR tangent",
        "const scaling_t *const RSTR scaling",
    ]
    params.extend(_abi_stream("h", component))
    params.extend(_abi_stream("out", component, const=""))
    entry_points.append(
        _AbiEntryPoint(
            name=inexact_apply_name(prefix, "compressed"),
            params=tuple(params),
            template="%s_impl" % inexact_apply_name(template_prefix, "compressed"),
            template_arguments=lambda scalar_type: "%s, compressed_t, scaling_t"
            % scalar_type,
        )
    )
    return tuple(entry_points)


def _c_abi_lines(prefix, dim, n_nodes, component, parameters, used_previous,
                 reads_state, packed_layouts=()):
    """`extern "C"` definitions, so the split is reachable from SFEM.

    The templated kernels above are what the generator produces; these are what
    the library links against.  One symbol each, carrying the scalar type as a
    width and its buffers as `void *` -- the shape `emitters/runtime_typed_abi`
    describes and the public dispatch above already had.
    """
    lines = []
    for entry_point in _c_abi_entry_points(
        prefix, dim, n_nodes, component, parameters, used_previous, reads_state,
        packed_layouts,
    ):
        lines.extend(
            _abi_entry_point(
                entry_point.name,
                list(entry_point.params),
                entry_point.template,
                entry_point.template_arguments,
            )
        )
    return lines


def _c_abi_prototype_lines(prefix, dim, n_nodes, component, parameters,
                           used_previous, reads_state, packed_layouts=()):
    """The same symbols, declared and not defined.

    What a delegating element publishes: the alias pass reads these to learn it
    has the entry points, then writes the definitions that forward to the twin.
    """
    lines = []
    for entry_point in _c_abi_entry_points(
        prefix, dim, n_nodes, component, parameters, used_previous, reads_state,
        packed_layouts,
    ):
        declared = runtime_typed_parameters(entry_point.params)
        lines.append('extern "C" int %s(' % entry_point.name)
        lines.extend(
            "    %s%s" % (param, "," if index + 1 < len(declared) else "")
            for index, param in enumerate(declared)
        )
        lines.extend([");", ""])
    return lines


#: The previous state reaches the ABI only where the material reads one.
_PREVIOUS_ABI_BY_USE = {
    True: lambda component: _abi_stream("z", component),
    False: lambda component: [],
}


def inexact_apply_files(material, unit, context):
    """The opt-in header for one unit, or ``()`` when the path does not apply.

    The header is named for the unit rather than the material, because a
    material with several units would otherwise have them collide on one path.

    An element whose mesh does not number its nodes lexicographically publishes
    no kernel of its own: `plans/layout.cartesian_twin` names the element whose
    micro-kernel it forwards to, and this emits the symbols and the reordering
    rather than a second body.
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
    return _FILES_BY_DELEGATION[
        cartesian_twin(context.element_type, dim, int(rule.n_shape)) is not None
    ](material, unit, context, dim, rule, flux_form, is_deformation_gradient)


def _own_kernel_files(
    material, unit, context, dim, rule, flux_form, is_deformation_gradient
):
    """An element that owns its micro-kernel: the header and its symbols."""
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
    stem = _inexact_stem(material, unit, context.element_type)
    return (
        ("%s_inline.hpp" % stem, header),
        ("%s_operator.cpp" % stem, operator_source),
    )


def _forwarded_kernel_files(
    material, unit, context, dim, rule, flux_form, is_deformation_gradient
):
    """An element that delegates publishes its symbols and no micro-kernel.

    The source it emits here is prototypes only.  `pipeline/driver.py`'s
    tensor-product alias pass rewrites it into definitions that forward to the
    twin's symbols with the connectivity reordered -- the same pass, and the
    same reordering, the energy path has always used; the declarations are what
    tell it this element has those entry points at all.

    Writing the forwarders here instead produced four duplicate symbols at link
    time, which is the shortest possible proof that the path was already there.
    """
    twin = cartesian_twin(context.element_type, dim, int(rule.n_shape))
    plan = emittable_inexact_apply_plan(
        twin.twin_name.upper(), dim, int(rule.n_shape), flux_form, rule
    )
    return _PROTOTYPES_BY_APPLICABILITY[plan is not None](
        plan, twin, material, unit, context, dim, rule, flux_form,
        is_deformation_gradient,
    )


def _delegated_prototype_files(
    plan, twin, material, unit, context, dim, rule, flux_form,
    is_deformation_gradient,
):
    """The symbols a delegating element publishes, where its twin has a plan."""
    unit_name = _unit_name(material, unit)
    prefix = "%s_%s" % (unit_name, str(context.element_type).lower())
    component = list(_COMPONENT_NAMES[:dim])
    adjugate, determinant = _adjugate_and_determinant(dim)
    integrand, gradient_symbols, previous_gradient_symbols = projected_tangent(
        plan, flux_form, rule, adjugate, determinant, is_deformation_gradient
    )
    lines = [
        '#include "../../op/sfem_%s_c_abi.hpp"' % material.op_name,
        "",
    ]
    lines.extend(
        _c_abi_prototype_lines(
            prefix, dim, plan.n_nodes, component,
            tuple(str(name) for name in flux_form.parameters),
            bool(plan.state_dependence(integrand, previous_gradient_symbols)),
            bool(plan.state_dependence(integrand, gradient_symbols)),
            tuple(l for l in plan.apply_layouts if l != "standard"),
        )
    )
    stem = _inexact_stem(material, unit, context.element_type)
    return ((("%s_operator.cpp" % stem), "\n".join(lines)),)


#: Whether the twin has a plan at all, answered by the plan layer and looked up
#: here -- the same table the owning path uses for the same question.
_PROTOTYPES_BY_APPLICABILITY = {
    True: _delegated_prototype_files,
    False: lambda plan, *arguments: (),
}

#: Whether this element owns its micro-kernel or forwards to a twin's.
_FILES_BY_DELEGATION = {False: _own_kernel_files, True: _forwarded_kernel_files}


def _inexact_stem(material, unit, element_type):
    return "%s_%s_inexact_apply" % (
        _unit_name(material, unit),
        str(element_type).lower(),
    )


def _unit_name(material, unit):
    """What this unit's kernels are called, matching the other emitters.

    `unit` is an emission kernel from `emission_kernels_for_context`, not a
    material sub-unit, and its name is already the composed output name --
    `conventions.unit_output_name` has run by the time it gets here.  Composing
    again doubles the material into `neohookean_ogden_neohookean_ogden_tet4`,
    which is what happens if this is mistaken for the driver's own spelling.
    """
    return str(getattr(unit, "name", None) or material.name)
