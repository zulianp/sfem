"""Assembling an element matrix without probing for its columns.

A Jacobian-action form is linear in the direction it is applied to: its flux
carries the direction only through the symbols `u<c>_direction_grad_<e>`.  So
the matrix column belonging to one trial degree of freedom is what the flux
becomes when that direction is the corresponding basis function -- which is a
substitution, made once at generation time, and not something a kernel has to
discover at run time by applying the operator to a unit vector and keeping the
answer.

That distinction is the whole point.  Probing costs one apply per trial degree
of freedom -- thirty per element on TET10 -- and each of those applies repeats
all of the state work.

What this gives is one *entry*, not a loop shape.  SFEM's own
`tet4_linear_elasticity_crs_adj` is the reference for what to do with the
entries once you have them: 144 of them, no loops, no quadrature, and 178
temporaries shared across every one -- sharing that exists only because the
entries are eliminated together, and that a loop over trial functions cannot
reach across.  So on a lowest-order simplex the caller builds every entry and
eliminates once; quadrature belongs where the element actually wants it.

`emitters/energy_codegen._sfem_soa_direct_hessian_matrix_assembly_lines` is not
the worked example.  It assembles without probing, which is better than the
residual path does, but it does so with a quadrature loop on an element whose
gradients are constant.  OP 28 records the comparison.

Here rather than in the emitter because it is a statement about the form, not
about how the form is spelled: the emitters ask for the substituted flux and
print whatever comes back.
"""

import sympy as sp

from codegen.framework.plans.evaluation_strategy import (
    EvaluationStrategy,
    evaluation_strategy,
)


#: Do not `expand` the flux on the way to the entries.
#:
#: Measured on the Mooney-Rivlin Kelvin-Voigt Newmark viscous flux: substituting
#: and then eliminating gives 483 temporaries and 2,980 operations for all 144
#: entries of the TET4 matrix.  Taking `sp.expand` first, to read the tangent
#: off as coefficients, gives **56,715,906** operations before elimination even
#: starts, and takes minutes.  Same mathematics, four orders of magnitude apart.
#:
#: The tangent is also dense -- all 81 of its `[row][d][col][e]` entries are
#: non-zero for this material -- so there is no structural sparsity waiting to
#: pay the expansion back.  Substitute, then let `cse` find the sharing.


def direction_gradient_symbols(field_names, dim):
    """The symbols a Jacobian-action flux carries its direction in.

    Indexed `[component][axis]`, matching the order the flux was built in.
    """
    return tuple(
        tuple(sp.Symbol("%s_direction_grad_%d" % (name, axis)) for axis in range(dim))
        for name in field_names
    )


def trial_direction_substitution(field_names, dim, trial_component, trial_gradient):
    """Replace the direction by one trial basis function.

    `trial_gradient[axis]` is whatever the emitter calls the trial function's
    physical gradient -- a name, because the shape it belongs to is a run-time
    loop index and only its gradient values differ between shapes.  Every
    component other than `trial_component` goes to zero, because a basis
    function for one field has no gradient in the others.
    """
    symbols = direction_gradient_symbols(field_names, dim)
    substitution = {}
    for component, row in enumerate(symbols):
        for axis, symbol in enumerate(row):
            substitution[symbol] = (
                trial_gradient[axis] if component == trial_component else sp.Integer(0)
            )
    return substitution


def flux_is_linear_in_the_direction(expression, field_names, dim):
    """Whether one flux component really is linear in the direction.

    The substitution is only a column of the matrix if it is.  A Jacobian action
    is linear by construction, so this is a statement about the form having been
    built as one -- worth asserting somewhere cheap rather than assuming, since
    a non-linear flux would substitute perfectly happily and give a wrong
    matrix with no other symptom.
    """
    symbols = [s for row in direction_gradient_symbols(field_names, dim) for s in row]
    present = [s for s in symbols if s in expression.free_symbols]
    for symbol in present:
        first = sp.diff(expression, symbol)
        if any(s in first.free_symbols for s in symbols):
            return False
    return True


def assembles_in_closed_form(element_type):
    """Whether this element's matrix is built with no loops at all.

    The same rule that decides how a form is *applied* decides how it is
    assembled, and for the same reason: it is a property of the element.  A
    lowest-order simplex has constant basis gradients and one quadrature point,
    so every entry of its matrix is an expression in the adjugate, the
    determinant and the state -- which is exactly the shape of SFEM's
    hand-written `tet4_linear_elasticity_crs_adj`, 144 entries and 178 shared
    temporaries with not one loop in it.

    Asking `plans.evaluation_strategy` rather than answering here is the point.
    The strategy is one table, and an assembly that made its own decision would
    be free to disagree with the apply it has to match.  A tensor-product
    element assembles through sum factorization and a higher-order simplex
    through quadrature, because that is what those elements want; neither is a
    lesser case of this one.
    """
    return evaluation_strategy(element_type) is EvaluationStrategy.EXPANDED


#: How many `[row][column]` entries a closed-form element matrix may hold before
#: this plan declines it.  The cost of the closed form is that `cse` sees every
#: entry at once, which is what buys the sharing and also what makes it
#: superlinear: TET4 with three fields is 144 entries, 483 temporaries and 2,980
#: operations, and TET10 with three fields would be 900 entries.  Lowest-order
#: simplices are the only elements `assembles_in_closed_form` admits, so nothing
#: reaches this limit today; it is here so that the day something does, it
#: declines visibly rather than by running for an hour.
CLOSED_FORM_ENTRY_LIMIT = 400


def closed_form_assembly_admits(
    *,
    element_type,
    tensor_product,
    matrix_formats,
    coefficients,
    dependencies,
    field_names,
    dim,
    reference_gradients,
    n_entries,
):
    """Whether this form on this element is assembled entry by entry.

    Every condition here is a fact about the form or the element rather than a
    preference, and all of them are asked in one place because an emitter that
    asked them itself would be emission deciding what to emit.

    The element must be one `assembles_in_closed_form` admits and must actually
    have constant reference gradients.  A matrix must be published at all, and
    it must not be a tensor-product element, whose assembly is factorised rather
    than enumerated.  The form must have a direction to substitute for, and must
    be linear in it -- otherwise substituting a trial function is not a matrix
    column, and the failure has no other symptom than a wrong number in the
    right place.  It must contract only test gradients, because a test *value*
    on a one-point rule needs the shape value at that point and the closed form
    folds no shape table.  And the matrix must be small enough to eliminate
    whole.
    """
    if reference_gradients is None:
        return False
    if tensor_product:
        return False
    if not assembles_in_closed_form(element_type):
        return False
    if not {"crs", "bsr"}.intersection(matrix_formats):
        return False
    if not dependencies.direction:
        return False
    if any(dependencies.value_coefficients):
        return False
    if not any(any(row) for row in dependencies.gradient_coefficients):
        return False
    if int(n_entries) > CLOSED_FORM_ENTRY_LIMIT:
        return False
    return all(
        flux_is_linear_in_the_direction(sp.sympify(expression), field_names, dim)
        for coefficient in coefficients
        for expression in coefficient.gradient
    )
