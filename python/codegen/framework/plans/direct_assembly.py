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
all of the state work.  Substituting instead lets the state work happen once per
quadrature point, with only the trial-dependent part inside the trial loop,
which is the shape the energy path already uses in
`emitters/energy_codegen._sfem_soa_direct_hessian_matrix_assembly_lines`.

Here rather than in the emitter because it is a statement about the form, not
about how the form is spelled: the emitters ask for the substituted flux and
print whatever comes back.
"""

import sympy as sp


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
