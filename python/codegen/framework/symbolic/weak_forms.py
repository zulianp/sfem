"""SoA weak-form lowering: energy density to gradient and Hessian-action forms.

This is form lowering -- differentiation of an energy against a deformation
gradient and assembly of the resulting weak coefficients.  It sits above the
symbolic objects it manipulates and below anything that knows about elements,
targets, or text.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import sympy as sp

from codegen.framework.symbolic.core import (
    _as_matrix,
    _check_scalar_expression,
    directional_derivative,
    matrix_inner,
)


@dataclass(frozen=True)
class SfemSoAKernelForm:
    name: str
    expression_graph: Optional["ExpressionGraph"] = None
    has_direction: bool = False
    output_mode: str = "accumulate"
    weak_form: Optional["SfemSoAWeakForm"] = None
    dependencies: object = None

    def __post_init__(self):
        if self.output_mode not in ("assign", "accumulate"):
            raise ValueError("output_mode must be 'assign' or 'accumulate'")
        if self.expression_graph is None and self.weak_form is None:
            raise ValueError("SfemSoAKernelForm requires expression_graph or weak_form")


def sfem_soa_kernel_form(
    name,
    expression_graph=None,
    has_direction=False,
    output_mode="accumulate",
    weak_form=None,
    dependencies=None,
):
    return SfemSoAKernelForm(name, expression_graph, has_direction, output_mode, weak_form, dependencies)


@dataclass(frozen=True)
class SfemSoAWeakForm:
    energy_density: sp.Expr
    deformation_gradient: Tuple[sp.Expr, ...]
    dim: int

    def __post_init__(self):
        dim = int(self.dim)
        deformation_gradient = tuple(self.deformation_gradient)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "energy_density", sp.sympify(self.energy_density))
        object.__setattr__(self, "deformation_gradient", deformation_gradient)
        if dim <= 0:
            raise ValueError("weak form dim must be positive")
        if len(deformation_gradient) != dim * dim:
            raise ValueError("deformation_gradient must have dim * dim entries")

    def deformation_gradient_matrix(self):
        return sp.Matrix(self.dim, self.dim, self.deformation_gradient)

    def first_piola(self):
        variables = self.deformation_gradient
        return sp.Matrix(
            self.dim,
            self.dim,
            [sp.diff(self.energy_density, variable) for variable in variables],
        )

    def linearized_first_piola(self, trial_gradient):
        trial_gradient = tuple(trial_gradient)
        if len(trial_gradient) != self.dim * self.dim:
            raise ValueError("trial_gradient must have dim * dim entries")
        P = self.first_piola()
        variables = self.deformation_gradient
        return sp.Matrix(
            self.dim,
            self.dim,
            [
                directional_derivative(P[i, j], variables, trial_gradient)
                for i in range(self.dim)
                for j in range(self.dim)
            ],
        )

    def diagnostic_expressions(self, has_direction=False):
        expressions = [self.energy_density]
        expressions.extend(tuple(self.first_piola()))
        if has_direction:
            trial_gradient = tuple(
                sp.symbols("trial_grad[%d]" % i)
                for i in range(self.dim * self.dim)
            )
            expressions.extend(tuple(self.linearized_first_piola(trial_gradient)))
        return tuple(expressions)


def sfem_soa_weak_form(energy_density, deformation_gradient):
    deformation_gradient = _as_matrix(deformation_gradient, "deformation_gradient")
    if deformation_gradient.shape[0] != deformation_gradient.shape[1]:
        raise ValueError("deformation_gradient must be square")
    return SfemSoAWeakForm(
        energy_density,
        tuple(deformation_gradient),
        deformation_gradient.shape[0],
    )
