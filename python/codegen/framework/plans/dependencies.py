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
