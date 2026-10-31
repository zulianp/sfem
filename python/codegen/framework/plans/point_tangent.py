"""The material's half of the stored tangent: the tangent at one point.

The inexact apply stores, at every point of the rule the exact operator
integrates with, the material tangent pulled back to the reference cell and
weighted (`docs/compressed_partial_assembly`):

    S^q_{ikmn} = w_q * A_{ijkl}(F(xi_q)) adj_{nj} adj_{ml} / det

and the apply contracts it with the increment's reference derivatives and the
test functions' -- a contraction that knows nothing of the material.  This plan
is everything that does: `A` at one point, differentiated from the form's flux
and written in that point's physical field gradients, the geometry's adjugate
and determinant, and the material's parameters.  The element's half -- which
points, which derivatives, how the numbers are grouped -- is
`plans/inexact_element.py`; the weight is the rule's and the library applies it.

There is no quadrature sum here and no projection.  The tangent at a point is
what the exact apply evaluates at that point; storing it instead of applying it
is the whole technique, so the stored apply is exact wherever the exact one is.

**Symmetry.**  An energy's flux is a gradient, so its tangent is a Hessian and
carries the major symmetry `S[i,k,m,n] == S[k,i,n,m]`: 45 numbers in three
dimensions.  A residual's is a Jacobian and generally does not -- the
Kelvin-Voigt viscous tangent genuinely does not -- and then all 81 are stored.
`symmetric` defaults to False because that is the safe direction to be wrong
in: 81 numbers for a symmetric tangent waste memory, 45 for an unsymmetric one
average away its antisymmetric part and return a different operator.

**The previous state is data.**  A rate-dependent material reads the previous
state's gradient too.  It is substituted, never differentiated against: the
tangent is the derivative with respect to the current state alone.
"""

from dataclasses import dataclass
import itertools
import re

import sympy as sp


@dataclass(frozen=True)
class PointTangentPlan:
    """The packing of one point's tangent."""

    dim: int
    symmetric: bool = False

    @property
    def tangent_components(self):
        order = self.dim * self.dim
        if self.symmetric:
            return order * (order + 1) // 2
        return order * order

    def tangent_index(self, i, k, m, n):
        """Where `S_{ikmn}` sits in the packed tangent.

        The symmetry is `S[i,k,m,n] == S[k,i,n,m]`, and getting it wrong is the
        easiest mistake in the whole construction.  It follows from what the
        indices are: `i` is the output component and `n` the test function's
        reference direction, `k` the increment's component and `m` its
        reference direction.  So the pair that swaps is `(i,n)` against `(k,m)`
        -- not `(i,k)` against `(m,n)`, which is the plausible-looking wrong
        answer and which no test comparing two packings of the same tensor can
        catch, because both would share it.  It is caught only by evaluating the
        action against a directly integrated one on a concrete element.

        Rows `r = i * dim + n`, columns `c = k * dim + m`: the symmetric tangent
        keeps the upper triangle of that matrix row by row
        (`plans/inexact_element.packed_slot`).
        """
        row, column = i * self.dim + n, k * self.dim + m
        order = self.dim * self.dim
        if not self.symmetric:
            return row * order + column
        if row > column:
            row, column = column, row
        return row * order - row * (row - 1) // 2 + (column - row)

    def state_dependence(self, tangent, state):
        """Which of `state`'s symbols the tangent actually reads.

        A state-independent material -- linear elasticity, whose tangent is the
        constant elasticity tensor -- reads none of them, and its set-up then
        gathers no state at all.
        """
        read = set()
        for expression in tangent:
            read |= expression.free_symbols
        return frozenset(symbol for row in state for symbol in row if symbol in read)


def flux_tangent_is_symmetric(flux_form):
    """Whether the flux's tangent carries the major symmetry `A[ijkl] == A[klij]`.

    Answered conservatively.  Only a difference that expands to exactly zero
    counts as symmetric; anything else is reported unsymmetric and costs memory
    rather than correctness.
    """
    dim = int(flux_form.dim)
    if int(flux_form.n_field_components) != dim:
        return False
    variables = list(flux_form.gradient)
    flux = list(flux_form.flux)
    tangent = {}
    for i, j, k, l in itertools.product(range(dim), repeat=4):
        tangent[(i, j, k, l)] = sp.diff(flux[i * dim + j], variables[k * dim + l])
    for i, j, k, l in itertools.product(range(dim), repeat=4):
        if sp.expand(tangent[(i, j, k, l)] - tangent[(k, l, i, j)]) != 0:
            return False
    return True


def point_tangent_plan(flux_forms):
    """The packing for the sum of these fluxes' tangents, or ``None``.

    A material with several units -- Mooney-Rivlin elasticity with
    Kelvin-Voigt viscosity -- stores one tangent, the sum of its units', so it is
    symmetric only when every unit's is.
    """
    flux_forms = tuple(flux_forms)
    if not flux_forms:
        return None
    dims = {int(form.dim) for form in flux_forms}
    if len(dims) != 1:
        return None
    dim = dims.pop()
    if any(int(form.n_field_components) != dim for form in flux_forms):
        return None
    return PointTangentPlan(
        dim=dim, symmetric=all(flux_tangent_is_symmetric(form) for form in flux_forms)
    )


def point_tangent_parameters(flux_forms):
    """The material constants the stored tangent reads, in the order it takes them.

    One order for the emitter and the operator that fills the array: sorted by
    name over the union of the units' parameters.
    """
    names = set()
    for form in flux_forms:
        names |= {str(parameter) for parameter in form.parameters}
    return tuple(sorted(names))


#: An integration-point field as the forms spell one of its components: `<name>_qp[<k>]`.
_QUADRATURE_COMPONENT = re.compile(r"^(\w+)_qp\[(\d+)\]$")


def point_tangent_quadrature_fields(flux_forms):
    """`((name, components), ...)`: the integration-point fields the fluxes read, by name.

    A relaxation group's history is one (`symbolic/relaxation.py`): data at each point, read
    by the tangent as it is by the exact kernels, and never differentiated against.
    """
    fields = {}
    for form in flux_forms:
        for expression in form.flux:
            for symbol in expression.free_symbols:
                match = _QUADRATURE_COMPONENT.match(str(symbol))
                if match:
                    name, component = match.group(1), int(match.group(2))
                    fields[name] = max(fields.get(name, 0), component + 1)
    return tuple(sorted(fields.items()))


def field_gradient_symbols(dim, role):
    """`grad u` at one point, named -- `gu_<component>_<axis>`."""
    return tuple(
        tuple(sp.Symbol("%s_%d_%d" % (role, component, axis)) for axis in range(dim))
        for component in range(dim)
    )


def reference_field_gradient_symbols(dim, role):
    """The same gradients before the geometry, on the reference cell."""
    return tuple(
        tuple(sp.Symbol("%s_ref_%d_%d" % (role, component, axis)) for axis in range(dim))
        for component in range(dim)
    )


def physical_field_gradient_definitions(
    reference_symbols, gradient_symbols, adjugate, determinant, dim
):
    """`du_c/dx_a = sum_m du_c/dxi_m * adj_{m,a} / det`, as (symbol, expression) pairs."""
    inverse = adjugate / determinant
    return [
        (
            gradient_symbols[component][axis],
            sum(reference_symbols[component][m] * inverse[m, axis] for m in range(dim)),
        )
        for component in range(dim)
        for axis in range(dim)
    ]


def geometry_symbols(dim):
    """The adjugate and determinant the tangent is written in."""
    return (
        sp.Matrix(dim, dim, lambda r, c: sp.Symbol("adjugate%d" % (r * dim + c))),
        sp.Symbol("determinant"),
    )


def point_tangent(plan, flux_form, adjugate, determinant, is_deformation_gradient):
    """One unit's pulled-back tangent at one point, unweighted.

    Returned in the point's physical field-gradient symbols (`gu_c_a`, and
    `gz_c_a` for the previous state), so that how they are computed stays the
    element's business: the set-up forms the reference gradient, pulls it
    through the geometry, and evaluates the material in those `dim * dim`
    values, as every other kernel in this framework does.

    Returns ``(tangent, gradient_symbols, previous_gradient_symbols)``.
    """
    dim = plan.dim
    variables = list(flux_form.gradient)
    flux = list(flux_form.flux)
    derivative = {
        (i, j, k, l): sp.diff(flux[i * dim + j], variables[k * dim + l])
        for i, j, k, l in itertools.product(range(dim), repeat=4)
    }
    # `is_deformation_gradient` says whether the form's variable is `I + grad u`
    # rather than `grad u`.  It shifts the diagonal and nothing else.
    shift = {True: sp.Integer(1), False: sp.Integer(0)}[bool(is_deformation_gradient)]
    gradient_symbols = field_gradient_symbols(dim, "gu")
    previous_gradient_symbols = field_gradient_symbols(dim, "gz")

    substitution = {}
    for c in range(dim):
        for axis in range(dim):
            substitution[variables[c * dim + axis]] = (
                shift * int(c == axis) + gradient_symbols[c][axis]
            )
    previous = tuple(flux_form.previous_gradient)
    if previous:
        for c in range(dim):
            for axis in range(dim):
                substitution[previous[c * dim + axis]] = previous_gradient_symbols[c][axis]

    tangent = [sp.Integer(0)] * plan.tangent_components
    seen = set()
    for i, k, m, n in itertools.product(range(dim), repeat=4):
        slot = plan.tangent_index(i, k, m, n)
        if slot in seen:
            continue
        seen.add(slot)
        pulled = (
            sum(
                derivative[(i, j, k, l)] * adjugate[n, j] * adjugate[m, l]
                for j, l in itertools.product(range(dim), repeat=2)
            )
            / determinant
        )
        tangent[slot] = pulled.subs(substitution)
    return tuple(tangent), gradient_symbols, previous_gradient_symbols


def flux_form_for_collection(collection, dim):
    """The flux form a lowered unit implies, or ``None`` when it implies none.

    Both front ends land here.  An energy unit differentiates its density into a
    flux; a residual unit differentiates the weak form against the test symbols,
    which is exact because a weak form is linear in its test function.

    Returns ``(flux_form, is_deformation_gradient)``.
    """
    from codegen.framework.forms.weak_forms import (
        flux_form_from_energy,
        flux_form_from_residual,
        sfem_soa_weak_form,
    )

    dim = int(dim)
    kind = getattr(getattr(collection, "kind", None), "value", None)
    if kind == "energy":
        variables = tuple(collection.variables)
        if not variables or len(variables) % dim:
            return None
        weak_form = sfem_soa_weak_form(
            collection.forms[0].expression,
            sp.Matrix(len(variables) // dim, dim, list(variables)),
        )
        return flux_form_from_energy(weak_form), weak_form.is_deformation_gradient
    if kind == "residual":
        expressions = tuple(getattr(collection, "residual_expressions", ()) or ())
        fields = tuple(getattr(collection, "residual_fields", ()) or ())
        if not expressions or len(expressions) != len(fields):
            return None
        # A residual differentiates against true test gradients, so its gradient
        # is `grad(u)` and never `I + grad(u)`.
        return flux_form_from_residual(expressions, fields, dim), False
    return None
