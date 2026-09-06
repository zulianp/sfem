"""The opt-in inexact apply, emitted as one self-contained kernel per element.

The plan is `plans/inexact_apply.py`; this prints it.  What it prints is fused
rather than split: the kernel takes the same arguments the exact apply takes,
builds the projected tangent from the state itself, and applies it, so the two
can be driven side by side by anything that can already drive the exact one.

Splitting the tangent into its own kernel and storing it per element is the
form that pays in a Krylov solve, where one tangent serves many applies.  It is
a different entry point with a different ABI and its own cache, and it is not
this.  Fusing first is what makes the comparison cheap: identical arguments,
identical driver, and on an affine simplex an answer that must agree.
"""

import itertools

import sympy as sp

from codegen.framework.emitters.cprinter import _sfem_ccode
from codegen.framework.plans.inexact_apply import (
    emittable_inexact_apply_plan,
    staged_action,
)
from codegen.framework.targets import current_target


def _reference_gradients(rule, n_nodes, dim):
    """The element's constant reference gradients, from the specialization."""
    values = tuple(rule.reference_gradients)
    if len(values) != n_nodes * dim:
        return None
    return tuple(
        tuple(sp.nsimplify(values[node * dim + direction]) for direction in range(dim))
        for node in range(n_nodes)
    )


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
    """One header defining `<material>_<element>_apply_inexact_affine_mesh_soa`.

    Returns ``None`` when the element or the form is not one this path covers:
    an element with no symbolic basis, a rule whose reference gradients are not
    constant, or a form whose flux is not differentiable into a tangent here.
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
    """The kernel itself, reached only when the plan layer said it applies."""
    gradients = _reference_gradients(rule, n_nodes, dim)

    component = ["x", "y", "z"][:dim]
    adjugate = sp.Matrix(
        dim, dim, lambda r, c: sp.Symbol("adjugate[%d]" % (r * dim + c))
    )
    determinant = sp.Symbol("determinant")
    inverse = adjugate / determinant

    state = [
        [sp.Symbol("u%s_%d" % (component[c], j)) for j in range(n_nodes)]
        for c in range(dim)
    ]
    increment = [
        [sp.Symbol("h%s_%d" % (component[c], j)) for j in range(n_nodes)]
        for c in range(dim)
    ]

    # The field gradient this element implies, in physical coordinates.
    physical = sp.zeros(dim, dim)
    for c in range(dim):
        for axis in range(dim):
            physical[c, axis] = sum(
                state[c][j]
                * sum(gradients[j][m] * inverse[m, axis] for m in range(dim))
                for j in range(n_nodes)
            )
    # `is_deformation_gradient` says whether the form's variable is `I + grad u`.
    variables = list(flux_form.gradient)
    substitution = {}
    for c in range(dim):
        for axis in range(dim):
            value = physical[c, axis]
            if is_deformation_gradient and c == axis:
                value = value + 1
            substitution[variables[c * dim + axis]] = value

    flux = list(flux_form.flux)
    tangent = {}
    for i, j, k, l in itertools.product(range(dim), repeat=4):
        tangent[(i, j, k, l)] = sp.diff(flux[i * dim + j], variables[k * dim + l])

    # Pull back and pack, by the symmetry `S[i,k,m,n] == S[k,i,n,m]`.
    packed = [None] * plan.tangent_components
    for i, k, m, n in itertools.product(range(dim), repeat=4):
        slot = plan.tangent_index(i, k, m, n)
        if packed[slot] is not None:
            continue
        packed[slot] = sum(
            tangent[(i, j, k, l)] * adjugate[n, j] * adjugate[m, l]
            for j, l in itertools.product(range(dim), repeat=2)
        ) / determinant

    tangent_symbols = [
        sp.Symbol("tangent%d" % slot) for slot in range(plan.tangent_components)
    ]
    output = [
        [sp.Symbol("element_out%d_%d" % (c, p)) for p in range(n_nodes)]
        for c in range(dim)
    ]
    stages = staged_action(plan, tangent_symbols, increment, output)

    body = []
    tangent_defs = [
        (symbol, expression.subs(substitution))
        for symbol, expression in zip(tangent_symbols, packed)
    ]
    body.extend(_assignment_lines(tangent_defs, "tangent"))
    for stage in stages:
        body.extend(_assignment_lines(stage.assignments, stage.name))

    parameters = tuple(str(name) for name in parameter_names)
    function = "%s_%s_apply_inexact_affine_mesh_soa" % (
        material_name,
        str(element_type).lower(),
    )
    return function, "\n".join(
        _kernel_lines(function, dim, n_nodes, component, parameters, body, output)
    )


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


def _assignment_lines(assignments, prefix):
    """Common subexpressions first, then the named values, as C declarations."""
    if not assignments:
        return []
    symbols = [symbol for symbol, _expression in assignments]
    expressions = [expression for _symbol, expression in assignments]
    temporaries, reduced = sp.cse(
        expressions, symbols=sp.numbered_symbols("%s_t" % prefix)
    )
    lines = [
        "            const scalar_t %s = %s;" % (symbol, _sfem_ccode(expression))
        for symbol, expression in temporaries
    ]
    lines.extend(
        "            const scalar_t %s = %s;" % (symbol, _sfem_ccode(expression))
        for symbol, expression in zip(symbols, reduced)
    )
    return lines


def _kernel_lines(function, dim, n_nodes, component, parameters, body, output):
    lines = [
        '#include "kernel_math.hpp"',
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t, typename jacobian_t>",
        "static SFEM_INLINE int %s_impl(" % function,
        "        const ptrdiff_t nelements,",
        "        const ptrdiff_t nnodes,",
        "        idx_t **const SFEM_RESTRICT elements,",
    ]
    lines.extend(
        "        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate%d," % index
        for index in range(dim * dim)
    )
    lines.append(
        "        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,"
    )
    lines.extend("        const scalar_t %s," % name for name in parameters)
    for role in ("u", "h"):
        lines.append("        const ptrdiff_t %s_stride," % role)
        lines.extend(
            "        const scalar_t *const SFEM_RESTRICT %s%s," % (role, name)
            for name in component
        )
    lines.append("        const ptrdiff_t out_stride,")
    lines.extend(
        "        scalar_t *const SFEM_RESTRICT out%s%s"
        % (name, "," if index + 1 < dim else "")
        for index, name in enumerate(component)
    )
    lines.extend(
        [
            ") {",
            "    (void)nnodes;",
            "",
            *_parallel_loop_lines(),
            "    for (ptrdiff_t element = 0; element < nelements; ++element) {",
            "        {",
        ]
    )
    lines.extend(
        "            const idx_t ev%d = elements[%d][element];" % (node, node)
        for node in range(n_nodes)
    )
    for name in component:
        for node in range(n_nodes):
            lines.append(
                "            const scalar_t u%s_%d = u%s[ev%d * u_stride];"
                % (name, node, name, node)
            )
            lines.append(
                "            const scalar_t h%s_%d = h%s[ev%d * h_stride];"
                % (name, node, name, node)
            )
    lines.append(
        "            const scalar_t adjugate[%d] = {%s};"
        % (
            dim * dim,
            ", ".join(
                "scalar_t(g_jacobian_adjugate%d[element])" % index
                for index in range(dim * dim)
            ),
        )
    )
    lines.append(
        "            const scalar_t determinant = scalar_t(g_jacobian_determinant0[element]);"
    )
    lines.extend(body)
    for index, name in enumerate(component):
        for node in range(n_nodes):
            lines.extend(
                _scatter_lines(
                    "out%s[ev%d * out_stride]" % (name, node),
                    "element_out%d_%d" % (index, node),
                    "            ",
                )
            )
    lines.extend(
        [
            "        }",
            "    }",
            "",
            "    return SFEM_SUCCESS;",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )
    return lines


def inexact_apply_files(material, unit, context):
    """The opt-in header for one element, or ``()`` when the path does not apply.

    Reads what it needs off the unit's lowered form collection, so it sees the
    same object every other emitter does rather than re-deriving the material.
    """
    from codegen.framework.symbolic.weak_forms import (
        flux_form_from_energy,
        sfem_soa_weak_form,
    )

    collection = getattr(unit, "form_collection", None)
    if collection is None or collection.kind.value != "energy":
        return ()
    variables = tuple(collection.variables)
    dim = int(unit.dim)
    if not variables or len(variables) % dim:
        return ()
    weak_form = sfem_soa_weak_form(
        collection.forms[0].expression,
        sp.Matrix(len(variables) // dim, dim, list(variables)),
    )
    flux_form = flux_form_from_energy(weak_form)
    rule = context.specialization.quadrature_rule
    emitted = inexact_apply_kernel_source(
        material.name,
        context.element_type,
        dim,
        int(rule.n_shape),
        flux_form,
        rule,
        flux_form.parameters,
        weak_form.is_deformation_gradient,
    )
    if emitted is None:
        return ()
    _function, source = emitted
    return (
        (
            "%s_%s_inexact_apply_inline.hpp"
            % (material.name, str(context.element_type).lower()),
            source,
        ),
    )
