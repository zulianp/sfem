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
