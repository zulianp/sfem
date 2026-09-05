"""Which form is being emitted, said as an order rather than as a name.

Both front ends already name their kernels the same way.  `add_energy` declares
``kernels=("objective", "gradient", "apply")`` and `add_residual` declares
``("gradient", "apply")``, and those names are the two formulations' words for
one sequence:

    objective   0-form   the energy, or a residual-based merit
    gradient    1-form   the gradient, or the negated residual
    apply       2-form   the Hessian action, or the Jacobian action

The emitters do not say it that way.  ``emitters/energy_codegen.py`` tests
``form.name == "objective"`` twenty times, ``"apply"`` eleven and
``"gradient"`` six, so the code reads as three special cases of a material
rather than three orders of one form.  While that is how it is written, the
energy and residual paths cannot converge: they would have to agree on strings
neither layer owns.

This is the vocabulary they can share.  It does not by itself remove the
branches -- an emitter asking ``form_order(form) is FormOrder.ZERO`` is still
asking -- but it makes the question one the form layer answers, which is the
precondition for answering it in a table instead.
"""

from enum import Enum

from codegen.framework.symbolic.forms import FormOrder

#: What each kernel name is, as an order.  The names come from the material's
#: `add_energy` / `add_residual` call and are the only three the framework has.
FORM_ORDER_BY_KERNEL = {
    "objective": FormOrder.ZERO,
    "gradient": FormOrder.ONE,
    "apply": FormOrder.TWO,
}


def form_order(form):
    """The order of the form this kernel emits.

    Raises for a name the framework does not define, rather than guessing: a
    new kernel name is a change to the form algebra and should be declared
    here, not inferred from a string comparison somewhere in emission.
    """
    name = str(getattr(form, "name", form))
    try:
        return FORM_ORDER_BY_KERNEL[name]
    except KeyError:
        raise ValueError(
            "no form order for kernel '%s'; the framework defines %s"
            % (name, ", ".join(sorted(FORM_ORDER_BY_KERNEL)))
        )


def writes_per_shape(form):
    """Whether this form scatters to shape functions.

    A 0-form accumulates a single value over the element; a 1-form and a 2-form
    both write one contribution per shape function.  That is the distinction
    every ``form.name != "objective"`` in emission is actually making, and it
    follows from the order rather than from the name.
    """
    return form_order(form) is not FormOrder.ZERO


class FormContraction(Enum):
    """How a form's integrand is contracted against the test functions."""

    #: The form carries a weak form: a strong flux is computed and contracted
    #: afterwards, which is what lets a tensor-product element sum-factorise it.
    DEFERRED_FLUX = "deferred_flux"

    #: The contraction is already done, pointwise, before emission sees it.
    POINTWISE = "pointwise"


def form_contraction(form):
    """Which contraction this form uses.

    Asked twenty-three times in the energy emitter as ``form.weak_form is
    None`` or ``is not None``, eight of them in one function.  It is a property
    of the form, and naming it is what lets the two paths be selected from a
    table rather than branched on.
    """
    return (
        FormContraction.POINTWISE
        if getattr(form, "weak_form", None) is None
        else FormContraction.DEFERRED_FLUX
    )


class FormReduction(Enum):
    """Where a 0-form's reduction to one scalar happens.

    Both formulations produce a 0-form, and below the form layer they differ in
    exactly one respect: whether the scalar is finished inside the element loop
    or after the scatter.

    ``ELEMENT_SUM`` is an energy or a recovered potential.  Each element
    contributes a number, the numbers are summed, and the total adds across
    operators -- which is what lets it be a term in the sum ``Function::value``
    accumulates.

    ``ASSEMBLED_NORM`` is ``1/2 * ||R||^2`` over the assembled residual, for a
    system that is the gradient of nothing.  It needs no element kernel of its
    own: the 1-form already computes R, so the whole reduction is one dot
    product over the degrees of freedom once the scatter is done.  It is not
    additive over operators, so it belongs to whoever holds the complete
    residual rather than to any one operator contributing part of it.
    """

    ELEMENT_SUM = "element_sum"
    ASSEMBLED_NORM = "assembled_norm"


#: The reduction each 0-form role implies.  A table rather than a chain of
#: comparisons, so emission reads the answer instead of deciding it.
FORM_REDUCTION_BY_ROLE = {
    "energy": FormReduction.ELEMENT_SUM,
    "potential": FormReduction.ELEMENT_SUM,
    "merit": FormReduction.ASSEMBLED_NORM,
}


def form_reduction(form):
    """How this 0-form reduces, from the role the form layer gave it.

    Raises for any other order: a 1-form and a 2-form scatter per shape and
    reduce nothing, so asking is a category error rather than a case to
    default.
    """
    # A lowered form states its own order; only a kernel name has to be looked
    # up in the table.  Both reach here, so prefer what the form already knows.
    order = getattr(form, "order", None)
    order = FormOrder(order) if order is not None else form_order(form)
    if order is not FormOrder.ZERO:
        raise ValueError(
            "only a 0-form has a reduction; %s is a %s"
            % (getattr(form, "name", form), order.name)
        )
    role = getattr(form, "role", None)
    role = getattr(role, "value", role)
    try:
        return FORM_REDUCTION_BY_ROLE[str(role)]
    except KeyError:
        raise ValueError(
            "no reduction defined for 0-form role '%s'; the framework defines %s"
            % (role, ", ".join(sorted(FORM_REDUCTION_BY_ROLE)))
        )


def objective_kernel_variants(form, emits_steps):
    """Which mesh kernels a form needs: the plain one, the stepped one, or both.

    A 0-form has two possible mesh kernels and only ever needs one of them.  The
    stepped kernel evaluates the form at `x + alpha * h` for a list of alphas,
    and the plain kernel evaluates it at `x`.  The second is the first with one
    alpha of zero: `x + 0 * h` is `x` exactly in IEEE arithmetic for any finite
    increment, and both then call the same block function.  So wherever the
    stepped kernel is emitted, the plain one is a duplicate of it.

    Emitting both is how `value` and `value_steps` came to disagree in the
    generated Op -- one zeroed its accumulator and the other did not -- which is
    the failure mode that having a single implementation removes.

    A 1-form and a 2-form scatter per shape and have no stepped variant, so they
    always take the plain kernel.
    """
    if writes_per_shape(form):
        return ("plain",)
    if emits_steps and getattr(form, "weak_form", None) is not None:
        return ("steps",)
    # A 0-form with no lowered weak form has no stepped kernel to be a
    # duplicate of -- the stepped emitter cannot build one -- so the plain
    # kernel is the only one and must still be emitted.  Getting this wrong
    # deletes the kernel outright rather than replacing it.
    return ("plain",)


def form_n_field_components(form, dim):
    """How many components the field this form acts on has.

    The generated kernels call this `DIM`, because for a displacement it equals
    the spatial dimension and every energy material the framework has is a
    displacement.  It is not the same number: it is the block size, the degrees
    of freedom per node, and a scalar field has one of them in any dimension.

    The lowered weak form knows it -- its flux has one row per component -- so
    the question is answered there when there is a weak form, and falls back to
    the spatial dimension when there is not, which is what every caller assumed
    unconditionally before.
    """
    weak_form = getattr(form, "weak_form", None)
    n_field_components = getattr(weak_form, "n_field_components", None)
    return dim if n_field_components is None else int(n_field_components)
