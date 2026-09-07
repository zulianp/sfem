"""The opt-in inexact apply, emitted as one header per element.

The plan is `plans/inexact_apply.py`; this prints it.

`Sbar` depends on the state and the geometry but not on the vector being
applied, so in a Krylov solve one tangent serves every apply of that Newton
step.  The header carries that split:

    <material>_<element>_inexact_apply_tangent_affine_mesh_soa
        once per tangent: state, geometry and material in, `Sbar` out

    <material>_<element>_inexact_apply_stored_affine_mesh_soa
    <material>_<element>_inexact_apply_compressed_affine_mesh_soa
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

import sympy as sp

from codegen.framework.emitters.cprinter import _sfem_ccode
from codegen.framework.plans.inexact_apply import (
    emittable_inexact_apply_plan,
    flux_form_for_collection,
    projected_tangent,
    staged_action,
)
from codegen.framework.targets import current_target

#: How the stored tangent is addressed.  Two strides rather than one so that
#: both layouts are expressible by the caller without a second kernel: element
#: major is `(components, 1)`, component major is `(1, nelements)`.
_TANGENT_ADDRESS = "element * tangent_element_stride + %d * tangent_component_stride"


def inexact_apply_kernel_source(
    material_name,
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
        element_type,
        dim,
        n_nodes,
        flux_form,
        rule,
        parameter_names,
        is_deformation_gradient,
    )


def _inexact_apply_kernel_source(
    plan,
    material_name,
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

    packed = projected_tangent(
        plan,
        flux_form,
        rule,
        adjugate,
        determinant,
        state,
        is_deformation_gradient,
        previous_state,
    )
    # Which state values the tangent actually reads.  A state-independent
    # material -- linear elasticity, whose tangent is constant -- reads none,
    # and then the kernel must not gather a state it will not use.  The plan
    # answers this; emission spells the answer.
    used_state = plan.state_dependence(packed, state)
    used_previous = plan.state_dependence(packed, previous_state)

    tangent_symbols = [
        sp.Symbol("tangent%d" % slot) for slot in range(plan.tangent_components)
    ]
    stages = staged_action(plan, tangent_symbols, increment, output)
    action_body = []
    for stage in stages:
        action_body.extend(_assignment_lines(stage.assignments, stage.name))

    parameters = tuple(str(name) for name in parameter_names)
    prefix = "%s_%s" % (material_name, str(element_type).lower())
    lines = [
        '#include "kernel_math.hpp"',
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    lines.extend(
        _tangent_lines(
            prefix, dim, n_nodes, component, parameters, used_state,
            used_previous, packed, plan,
        )
    )
    lines.extend(_stored_lines(prefix, n_nodes, component, plan, action_body))
    lines.extend(_compressed_lines(prefix, n_nodes, component, plan, action_body))
    lines.extend(["} // namespace codegen", "} // namespace sfem", ""])
    return "%s_inexact_apply_tangent_affine_mesh_soa" % prefix, "\n".join(lines)


#: Whether the plan layer produced a plan decides whether there is a kernel.
#: A table, for the reason the energy emitter's bodies are ones: emission looks
#: the answer up rather than deciding it again.
_KERNEL_BY_APPLICABILITY = {
    True: lambda plan, *arguments: _inexact_apply_kernel_source(plan, *arguments),
    False: lambda plan, *arguments: None,
}


def _parallel_loop_lines():
    """The pragma opening the element loop, from the bound target."""
    target = current_target()
    if target is None or not hasattr(target, "parallel_element_loop_lines"):
        return []
    return ["    %s" % line for line in target.parallel_element_loop_lines("static")]


def _scatter_lines(lhs, rhs, indent):
    """A scatter-add spelled by the bound target rather than by a literal."""
    target = current_target()
    if target is None or not hasattr(target, "scatter_add_lines"):
        return ["%s%s += %s;" % (indent, lhs, rhs)]
    return list(target.scatter_add_lines(lhs, rhs, indent))


def _assignment_lines(assignments, prefix, indent="        "):
    """Common subexpressions first, then the named values, as C declarations."""
    if not assignments:
        return []
    symbols = [symbol for symbol, _expression in assignments]
    expressions = [expression for _symbol, expression in assignments]
    temporaries, reduced = sp.cse(
        expressions, symbols=sp.numbered_symbols("%s_t" % prefix)
    )
    lines = [
        "%sconst scalar_t %s = %s;" % (indent, symbol, _sfem_ccode(expression))
        for symbol, expression in temporaries
    ]
    lines.extend(
        "%sconst scalar_t %s = %s;" % (indent, symbol, _sfem_ccode(expression))
        for symbol, expression in zip(symbols, reduced)
    )
    return lines


def _element_lines(n_nodes, indent="        "):
    return [
        "%sconst idx_t ev%d = elements[%d][element];" % (indent, node, node)
        for node in range(n_nodes)
    ]


def _gather_lines(role, component, n_nodes, wanted, indent="        "):
    """Element gathers for one role, restricted to the values that are read."""
    return [
        "%sconst scalar_t %s%s_%d = %s%s[ev%d * %s_stride];"
        % (indent, role, name, node, role, name, node, role)
        for name in component
        for node in range(n_nodes)
        if sp.Symbol("%s%s_%d" % (role, name, node)) in wanted
    ]


def _geometry_lines(dim, indent="        "):
    lines = [
        "%sconst scalar_t adjugate%d = scalar_t(g_jacobian_adjugate%d[element]);"
        % (indent, index, index)
        for index in range(dim * dim)
    ]
    lines.append(
        "%sconst scalar_t determinant = scalar_t(g_jacobian_determinant0[element]);"
        % indent
    )
    return lines


def _geometry_arguments(dim):
    lines = [
        "        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate%d," % index
        for index in range(dim * dim)
    ]
    lines.append("        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,")
    return lines


def _stream_arguments(role, component):
    lines = ["        const ptrdiff_t %s_stride," % role]
    lines.extend(
        "        const scalar_t *const SFEM_RESTRICT %s%s," % (role, name)
        for name in component
    )
    return lines


def _output_arguments(component):
    lines = ["        const ptrdiff_t out_stride,"]
    lines.extend(
        "        scalar_t *const SFEM_RESTRICT out%s%s"
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
                    "        ",
                )
            )
    return lines


def _tangent_lines(
    prefix, dim, n_nodes, component, parameters, used_state, used_previous,
    packed, plan,
):
    """The partial assembly: `Sbar` computed once and stored.

    Takes the previous state only when the material reads one, so a
    rate-independent material keeps the shorter signature.
    """
    tangent_symbols = [
        sp.Symbol("tangent%d" % slot) for slot in range(plan.tangent_components)
    ]
    body = _element_lines(n_nodes)
    body.extend(_gather_lines("u", component, n_nodes, used_state))
    body.extend(_gather_lines("z", component, n_nodes, used_previous))
    body.extend(_geometry_lines(dim))
    body.extend(_assignment_lines(list(zip(tangent_symbols, packed)), "tangent"))
    body.extend(
        "        tangent[%s] = tangent_t(tangent%d);" % (_TANGENT_ADDRESS % slot, slot)
        for slot in range(plan.tangent_components)
    )

    signature = ["        const ptrdiff_t nelements,", "        idx_t **const SFEM_RESTRICT elements,"]
    signature.extend(_geometry_arguments(dim))
    signature.extend("        const scalar_t %s," % name for name in parameters)
    signature.extend(_stream_arguments("u", component))
    signature.extend(_PREVIOUS_STREAMS_BY_USE[bool(used_previous)](component))
    signature.extend(
        [
            "        const ptrdiff_t tangent_element_stride,",
            "        const ptrdiff_t tangent_component_stride,",
            "        tangent_t *const SFEM_RESTRICT tangent",
        ]
    )
    return _function_lines(
        "%s_inexact_apply_tangent_affine_mesh_soa" % prefix,
        "template <typename scalar_t, typename jacobian_t, typename tangent_t>",
        signature,
        body,
    )


def _stored_lines(prefix, n_nodes, component, plan, action_body):
    """The apply: stored tangent and the vector, and nothing else."""
    body = _element_lines(n_nodes)
    body.extend(_gather_lines("h", component, n_nodes, _all_names("h", component, n_nodes)))
    body.extend(
        "        const scalar_t tangent%d = scalar_t(tangent[%s]);"
        % (slot, _TANGENT_ADDRESS % slot)
        for slot in range(plan.tangent_components)
    )
    body.extend(action_body)
    body.extend(_scatter_body(component, n_nodes))

    signature = ["        const ptrdiff_t nelements,", "        idx_t **const SFEM_RESTRICT elements,"]
    signature.extend(
        [
            "        const ptrdiff_t tangent_element_stride,",
            "        const ptrdiff_t tangent_component_stride,",
            "        const tangent_t *const SFEM_RESTRICT tangent,",
        ]
    )
    signature.extend(_stream_arguments("h", component))
    signature.extend(_output_arguments(component))
    return _function_lines(
        "%s_inexact_apply_stored_affine_mesh_soa" % prefix,
        "template <typename scalar_t, typename tangent_t>",
        signature,
        body,
    )


def _compressed_lines(prefix, n_nodes, component, plan, action_body):
    """The same apply from a scaled low-precision store.

    The scale multiplies the outputs, not the tangent: the action is linear in
    `Sbar`, so it is the same number either way, and there are `dim * n_nodes`
    outputs against the tangent's 45 components.
    """
    body = _element_lines(n_nodes)
    body.extend(_gather_lines("h", component, n_nodes, _all_names("h", component, n_nodes)))
    body.append("        const scalar_t scale = scalar_t(scaling[element]);")
    body.extend(
        "        const scalar_t tangent%d = scalar_t(tangent[%s]);"
        % (slot, _TANGENT_ADDRESS % slot)
        for slot in range(plan.tangent_components)
    )
    body.extend(action_body)
    body.extend(_scatter_body(component, n_nodes, scale="scale * "))

    signature = ["        const ptrdiff_t nelements,", "        idx_t **const SFEM_RESTRICT elements,"]
    signature.extend(
        [
            "        const ptrdiff_t tangent_element_stride,",
            "        const ptrdiff_t tangent_component_stride,",
            "        const tangent_t *const SFEM_RESTRICT tangent,",
            "        const scale_t *const SFEM_RESTRICT scaling,",
        ]
    )
    signature.extend(_stream_arguments("h", component))
    signature.extend(_output_arguments(component))
    return _function_lines(
        "%s_inexact_apply_compressed_affine_mesh_soa" % prefix,
        "template <typename scalar_t, typename tangent_t, typename scale_t>",
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


def _all_names(role, component, n_nodes):
    return frozenset(
        sp.Symbol("%s%s_%d" % (role, name, node))
        for name in component
        for node in range(n_nodes)
    )


def _function_lines(name, template, signature, body):
    lines = [template, "static SFEM_INLINE int %s_impl(" % name]
    lines.extend(signature)
    lines.append(") {")
    lines.extend(_parallel_loop_lines())
    lines.append("    for (ptrdiff_t element = 0; element < nelements; ++element) {")
    lines.extend(body)
    lines.extend(["    }", "", "    return SFEM_SUCCESS;", "}", ""])
    return lines


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
    _function, source = emitted
    return (
        (
            "%s_%s_inexact_apply_inline.hpp"
            % (_unit_name(material, unit), str(context.element_type).lower()),
            source,
        ),
    )


def _unit_name(material, unit):
    """What this unit's kernels are called, matching the other emitters."""
    return str(getattr(unit, "name", None) or material.name)
