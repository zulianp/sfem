"""Dependency pruning: what a kernel actually reads, and what is dead.

Given the weak coefficients of a form, this decides which field roles the kernel
touches, whether it touches values or gradients of each, which material
parameters survive, and which individual coefficient entries are structurally
zero.  Everything downstream keys off the answer: absent roles produce no
argument, no gather and no memory traffic, and zero coefficients drop whole
terms out of the emitted loops.

That is a planning decision -- the PRD calls it dependency-pruned form
collections -- and it is pure analysis of free symbols.  It was being computed
inside ``emitters/residual_codegen.py``, which meant an emitter was doing
symbolic analysis to work out what it should emit.  It answers the same question
as ``symbolic.residual.residual_dependencies_for`` and additionally reports
per-coefficient structural zeros, which is what the emitter needs to skip terms.

``system`` here is a ``ResidualEmissionModel``: the eight-member record the
planning layer builds from a lowered form collection.
"""

from dataclasses import dataclass, replace

import sympy as sp


def _is_zero(expression):
    return sp.sympify(expression) == 0


@dataclass(frozen=True)
class ResidualCodegenDependencies:
    current: bool
    previous: bool
    direction: bool
    parameters: tuple
    current_value: bool
    current_gradient: bool
    previous_value: bool
    previous_gradient: bool
    direction_value: bool
    direction_gradient: bool
    value_coefficients: tuple
    gradient_coefficients: tuple
    #: The symbols the coefficients actually read, per role.  The booleans
    #: above are `any(...)` over the fields; these keep the detail the
    #: reduction throws away, so a caller can ask what *one* field
    #: contributes rather than what the system does -- see
    #: `plans.streams.field_stream_usage`, and the block form it exists for.
    current_symbols: tuple = ()
    previous_symbols: tuple = ()
    direction_symbols: tuple = ()

    @property
    def uses_trial_gradients(self):
        return self.current_gradient or self.previous_gradient or self.direction_gradient

    @property
    def uses_test_gradients(self):
        return any(any(row) for row in self.gradient_coefficients)

    @property
    def uses_test_coefficients(self):
        return any(self.value_coefficients) or self.uses_test_gradients

    @property
    def uses_reference_gradients(self):
        return self.uses_trial_gradients or self.uses_test_gradients

    @property
    def uses_adjugate(self):
        return self.uses_reference_gradients


#: The order the test function's quantities are declared in, which is the order
#: the generated contraction bodies have always used.
TEST_QUANTITY_ORDER = ("value", "gradient")


def live_test_coefficients(dependencies, row, dim):
    """One field row's live coefficients, as ``(kind, axis, name)``.

    A coefficient multiplies the test function's value or one of its
    derivatives, and ``TEST_QUANTITY_ORDER`` above is the order the two are
    declared in.  Which of a row's coefficients exist follows from the lowered
    form: a structurally zero one contributes nothing, and is neither staged nor
    assigned nor contracted.

    The gradient half of that was already a comprehension filter at four sites
    in ``emitters/residual_codegen.py`` -- an absent coefficient simply not
    appearing in the sequence -- while the value half was an ``if`` beside it
    saying the same thing a different way.  One sequence says both, and a row
    with nothing live yields nothing rather than being skipped by a test.

    ``kind`` is ``TEST_QUANTITY_ORDER``'s word and ``axis`` is the derivative's
    direction, or ``None`` for the value.  What the coefficient is *multiplied
    by* stays with the caller: the local bodies spell the test factor one way
    and the mixed ones another, with the field and the test index in the name.
    """
    coefficients = []
    if dependencies.value_coefficients[row]:
        coefficients.append(("value", None, "value_coeff%d" % row))
    coefficients.extend(
        ("gradient", axis, "grad_coeff%d_%d" % (row, axis))
        for axis in range(dim)
        if dependencies.gradient_coefficients[row][axis]
    )
    return tuple(coefficients)


def publishes_kernel(dependencies):
    """Whether this form contributes anything, and so has a kernel at all.

    A block of a coupled system whose coefficients are all structurally zero
    contracts nothing onto the test functions.  The kernel for it is an empty
    loop nest over quadrature points, test functions and lanes, or a mesh
    entry point whose whole body is `return SFEM_SUCCESS;` -- and an empty loop
    nest is not a kernel, it is the absence of one written down.

    So the answer here is not "emit a kernel that does nothing"; it is that
    there is nothing to emit, and the local block, the element entry point, the
    mesh kernels, their diagnostics record and their C ABI entries all follow
    from that.
    """
    return dependencies.uses_test_coefficients


def contracted_gradient_components(dependencies, dim):
    """The reference-gradient components a form's test functions contract with.

    Empty when the form contracts only values.  This is the same question
    `uses_test_gradients` answers, phrased as a range so that emission can
    iterate it instead of branching on it -- the decision is the plan's.
    """
    return tuple(range(dim)) if dependencies.uses_test_gradients else ()


def contracted_test_quantities(dependencies):
    """Which of the test function's value and gradient the form contracts.

    The same shape as `plans.geometry_quantities.local_geometry_quantities`,
    and for the same reason: a form that contracts only test gradients still
    opened its contraction with a shape lookup for a value it never read, and
    the fix is for emission to iterate what the plan returns rather than to
    branch on the plan itself.  `uses_test_coefficients` above is the
    disjunction of these two; this keeps them apart, which is what a
    declaration needs.
    """
    quantities = []
    if any(dependencies.value_coefficients):
        quantities.append("value")
    if dependencies.uses_test_gradients:
        quantities.append("gradient")
    return tuple(quantities)


def substituted_trial_quantities(dependencies):
    """Which of the trial function's value and gradient an assembly substitutes.

    The mirror of `contracted_test_quantities`, on the other side of the
    bilinear form, and returned as a sequence for the same reason: emission
    iterates what the plan says rather than branching on the plan itself.

    A matrix column is the flux with a trial basis function put where the
    direction was, so what has to be substituted is exactly what the flux reads
    the direction through.  A gradient is always among them -- a form with no
    gradient coefficient publishes no assembly at all -- and a value only when
    the flux contracts one.
    """
    quantities = []
    if dependencies.direction_value:
        quantities.append("value")
    if dependencies.direction_gradient:
        quantities.append("gradient")
    return tuple(quantities)


def live_gradient_directions(dependencies, dim):
    """The spatial directions in which some row's gradient coefficient is live.

    `contracted_gradient_components` above answers the coarser question -- does
    this form contract test gradients at all -- as a range, and says in its own
    docstring that it is phrased that way so emission can iterate instead of
    branching.  This is the same shape one step finer.  A direction in which
    every row's coefficient is structurally zero contributes nothing: it is
    neither declared nor contracted, and it should simply not appear in the
    sequence.

    Two sites in `emitters/residual_codegen.py` spelled that as a range with a
    skip inside it -- `for d in range(dim)` guarded by `if not any(row[d] for
    row in dependencies.gradient_coefficients): continue` -- which is the
    sequence written as its own complement.  `live_test_coefficients` below
    already handles the per-row case this way.
    """
    return tuple(
        d
        for d in range(dim)
        if any(row[d] for row in dependencies.gradient_coefficients)
    )


def staged_test_quantities(dependencies):
    """Which per-field coefficient buffers a tensor-product body stages.

    Deliberately not the same sequence as `contracted_test_quantities` above.
    That one says which quantities the form *contracts*, and drops the value
    when no coefficient multiplies it.  This says which buffers exist, and the
    value buffer always does: the contraction reads it whether or not a
    coefficient is live, so a row without one stages a zero rather than the
    kernel taking a second shape.  `_TENSOR_INTEGRATE_BY_QUANTITIES` in
    `emitters/residual_codegen.py` already records that from the other side, in
    a comment beside its `("gradient",)` entry.

    Three sites in that emitter asked `dependencies.uses_test_gradients`
    instead: the buffer declaration, the assignment inside the quadrature loop,
    and the contraction call that reads it.  That is one question about what is
    staged, asked separately by the code that declares the buffer, the code
    that fills it and the code that consumes it -- three places to disagree
    about a buffer's existence.
    """
    if dependencies.uses_test_gradients:
        return ("value", "gradient")
    return ("value",)


def residual_codegen_dependencies(system, coefficients, dependencies):
    free_symbols = set()
    for coefficient in coefficients:
        free_symbols.update(sp.sympify(coefficient.value).free_symbols)
        for expression in coefficient.gradient:
            free_symbols.update(sp.sympify(expression).free_symbols)
    candidate_parameters = tuple(
        dict.fromkeys(tuple(dependencies.parameters) + tuple(system.parameters))
    )
    current_value = any(field.value in free_symbols for field in system.fields)
    current_gradient = any(
        free_symbols.intersection(field.gradient) for field in system.fields
    )
    previous_value = any(
        field.previous_value is not None and field.previous_value in free_symbols
        for field in system.fields
    )
    previous_gradient = any(
        free_symbols.intersection(field.previous_gradient) for field in system.fields
    )
    direction_value = any(
        field.direction_value in free_symbols for field in system.fields
    )
    direction_gradient = any(
        free_symbols.intersection(field.direction_gradient) for field in system.fields
    )
    return ResidualCodegenDependencies(
        current=current_value or current_gradient,
        previous=previous_value or previous_gradient,
        direction=direction_value or direction_gradient,
        parameters=tuple(
            parameter for parameter in candidate_parameters if parameter in free_symbols
        ),
        current_value=current_value,
        current_gradient=current_gradient,
        previous_value=previous_value,
        previous_gradient=previous_gradient,
        direction_value=direction_value,
        direction_gradient=direction_gradient,
        value_coefficients=tuple(
            not _is_zero(coefficient.value) for coefficient in coefficients
        ),
        gradient_coefficients=tuple(
            tuple(not _is_zero(expression) for expression in coefficient.gradient)
            for coefficient in coefficients
        ),
        current_symbols=tuple(
            symbol
            for field in system.fields
            for symbol in field.current_symbols
            if symbol in free_symbols
        ),
        previous_symbols=tuple(
            symbol
            for field in system.fields
            for symbol in field.previous_symbols
            if symbol in free_symbols
        ),
        direction_symbols=tuple(
            symbol
            for field in system.fields
            for symbol in field.direction_symbols
            if symbol in free_symbols
        ),
    )


def assembled_matrix_dependencies(dependencies):
    """The dependency view an assembled matrix kernel sees.

    A matrix is assembled, not applied, so there is no direction to apply it
    to: the direction role and both of its access flags are dead regardless of
    what the form's free symbols say.  Three assembly emitters were each
    rebuilding the whole twelve-field record by hand to state that, in three
    byte-identical copies.
    """
    return replace(
        dependencies,
        direction=False,
        direction_value=False,
        direction_gradient=False,
    )


def jacobian_action_dependencies(dependencies):
    """The dependency view a Jacobian-action kernel sees.

    The mirror of the assembled view.  An action is applied *to* a direction,
    so the direction role is live at the kernel boundary whether or not the
    form mentions it, and the emitters said so by appending the direction
    stride and pointers unconditionally -- next to conditional chains for every
    other role, which left the signature and the dependency set disagreeing
    about the same kernel.

    Only the role is forced.  ``direction_value`` and ``direction_gradient``
    say what the body reads once the direction is in hand, which is the form's
    decision and not the boundary's.
    """
    return replace(dependencies, direction=True)
