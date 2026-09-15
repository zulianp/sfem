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
