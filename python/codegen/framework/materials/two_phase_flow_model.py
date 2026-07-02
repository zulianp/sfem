from dataclasses import dataclass
from typing import Tuple

import sympy as sp

from ..symbolic.constitutive import (
    TwoPhaseFlowConstitutiveModel,
    TwoPhaseFlowConstitutiveState,
)
from ..symbolic.core import KernelExpressions
from ..symbolic.core import directional_derivative


@dataclass(frozen=True)
class TwoPhaseFlowImplicitEulerState:
    current: TwoPhaseFlowConstitutiveState
    previous: TwoPhaseFlowConstitutiveState
    water_mass_current: sp.Expr
    water_mass_previous: sp.Expr
    co2_mass_current: sp.Expr
    co2_mass_previous: sp.Expr
    water_accumulation: sp.Expr
    co2_accumulation: sp.Expr
    water_mass_flux: Tuple[sp.Expr, ...]
    co2_mass_flux: Tuple[sp.Expr, ...]
    water_residual: sp.Expr
    co2_residual: sp.Expr

    @property
    def residual(self):
        return sp.Matrix((self.water_residual, self.co2_residual))


@dataclass(frozen=True)
class TwoPhaseFlowJacobianAction:
    ww: sp.Expr
    wc: sp.Expr
    cw: sp.Expr
    cc: sp.Expr
    water_action: sp.Expr
    co2_action: sp.Expr
    merit: sp.Expr

    @property
    def action(self):
        return sp.Matrix((self.water_action, self.co2_action))

    @property
    def blocks(self):
        return ((self.ww, self.wc), (self.cw, self.cc))


@dataclass(frozen=True)
class TwoPhaseFlowImplicitEulerModel:
    constitutive: TwoPhaseFlowConstitutiveModel
    dim: int

    def __post_init__(self):
        dim = int(self.dim)
        if dim not in (1, 2, 3):
            raise ValueError("two-phase flow dimension must be 1, 2, or 3")
        object.__setattr__(self, "dim", dim)

    @classmethod
    def symbolic(cls, dim, prefix=""):
        return cls(TwoPhaseFlowConstitutiveModel.symbolic(prefix), dim)

    def weak_residual(
        self,
        water_pressure,
        co2_pressure,
        water_pressure_old,
        co2_pressure_old,
        water_pressure_gradient,
        co2_pressure_gradient,
        water_test_value,
        co2_test_value,
        water_test_gradient,
        co2_test_gradient,
        intrinsic_permeability,
        dt,
    ):
        grad_pw = _as_vector(water_pressure_gradient, self.dim, "water_pressure_gradient")
        grad_pc = _as_vector(co2_pressure_gradient, self.dim, "co2_pressure_gradient")
        grad_vw = _as_vector(water_test_gradient, self.dim, "water_test_gradient")
        grad_vc = _as_vector(co2_test_gradient, self.dim, "co2_test_gradient")
        permeability = _as_square_matrix(
            intrinsic_permeability,
            self.dim,
            "intrinsic_permeability",
        )
        dt = sp.sympify(dt)
        if dt.is_number and dt <= 0:
            raise ValueError("dt must be positive")

        current = self.constitutive.state(water_pressure, co2_pressure)
        previous = self.constitutive.state(water_pressure_old, co2_pressure_old)
        porosity = self.constitutive.parameters.porosity

        water_mass_current = porosity * current.water_saturation * current.water_density
        water_mass_previous = (
            porosity * previous.water_saturation * previous.water_density
        )
        co2_mass_current = porosity * current.co2_saturation * current.co2_density
        co2_mass_previous = porosity * previous.co2_saturation * previous.co2_density
        water_accumulation = (water_mass_current - water_mass_previous) / dt
        co2_accumulation = (co2_mass_current - co2_mass_previous) / dt

        water_mass_flux = -(
            current.water_density
            * current.water_mobility
            * permeability
            * grad_pw
        )
        co2_mass_flux = -(
            current.co2_density
            * current.co2_mobility
            * permeability
            * grad_pc
        )
        water_residual = (
            water_accumulation * sp.sympify(water_test_value)
            - water_mass_flux.dot(grad_vw)
        )
        co2_residual = (
            co2_accumulation * sp.sympify(co2_test_value)
            - co2_mass_flux.dot(grad_vc)
        )

        return TwoPhaseFlowImplicitEulerState(
            current=current,
            previous=previous,
            water_mass_current=water_mass_current,
            water_mass_previous=water_mass_previous,
            co2_mass_current=co2_mass_current,
            co2_mass_previous=co2_mass_previous,
            water_accumulation=water_accumulation,
            co2_accumulation=co2_accumulation,
            water_mass_flux=tuple(water_mass_flux),
            co2_mass_flux=tuple(co2_mass_flux),
            water_residual=water_residual,
            co2_residual=co2_residual,
        )

    def kernel_expressions(self, *args, **kwargs):
        state = self.weak_residual(*args, **kwargs)
        return (
            KernelExpressions()
            .residual(state.water_residual, "water_residual")
            .residual(state.co2_residual, "co2_residual")
        )

    def linearized_weak_residual(
        self,
        water_pressure,
        co2_pressure,
        water_pressure_old,
        co2_pressure_old,
        water_pressure_gradient,
        co2_pressure_gradient,
        water_direction,
        co2_direction,
        water_direction_gradient,
        co2_direction_gradient,
        water_test_value,
        co2_test_value,
        water_test_gradient,
        co2_test_gradient,
        intrinsic_permeability,
        dt,
    ):
        state = self.weak_residual(
            water_pressure,
            co2_pressure,
            water_pressure_old,
            co2_pressure_old,
            water_pressure_gradient,
            co2_pressure_gradient,
            water_test_value,
            co2_test_value,
            water_test_gradient,
            co2_test_gradient,
            intrinsic_permeability,
            dt,
        )
        grad_pw = _as_vector(water_pressure_gradient, self.dim, "water_pressure_gradient")
        grad_pc = _as_vector(co2_pressure_gradient, self.dim, "co2_pressure_gradient")
        grad_hw = _as_vector(
            water_direction_gradient,
            self.dim,
            "water_direction_gradient",
        )
        grad_hc = _as_vector(
            co2_direction_gradient,
            self.dim,
            "co2_direction_gradient",
        )
        water_variables = (sp.sympify(water_pressure),) + tuple(grad_pw)
        co2_variables = (sp.sympify(co2_pressure),) + tuple(grad_pc)
        water_directions = (sp.sympify(water_direction),) + tuple(grad_hw)
        co2_directions = (sp.sympify(co2_direction),) + tuple(grad_hc)

        ww = directional_derivative(
            state.water_residual,
            water_variables,
            water_directions,
        )
        wc = directional_derivative(
            state.water_residual,
            co2_variables,
            co2_directions,
        )
        cw = directional_derivative(
            state.co2_residual,
            water_variables,
            water_directions,
        )
        cc = directional_derivative(
            state.co2_residual,
            co2_variables,
            co2_directions,
        )
        return TwoPhaseFlowJacobianAction(
            ww=ww,
            wc=wc,
            cw=cw,
            cc=cc,
            water_action=ww + wc,
            co2_action=cw + cc,
            merit=sp.Rational(1, 2)
            * (
                state.water_residual * state.water_residual
                + state.co2_residual * state.co2_residual
            ),
        )

    def linearized_kernel_expressions(self, *args, **kwargs):
        linearization = self.linearized_weak_residual(*args, **kwargs)
        return (
            KernelExpressions()
            .jacobian_action(linearization.ww, "jacobian_ww")
            .jacobian_action(linearization.wc, "jacobian_wc")
            .jacobian_action(linearization.cw, "jacobian_cw")
            .jacobian_action(linearization.cc, "jacobian_cc")
            .jacobian_action(linearization.water_action, "water_jacobian_action")
            .jacobian_action(linearization.co2_action, "co2_jacobian_action")
            .merit(linearization.merit, "residual_norm_merit")
        )

    def build_linearized_expression_graph(
        self,
        water_pressure,
        co2_pressure,
        water_pressure_old,
        co2_pressure_old,
        water_pressure_gradient,
        co2_pressure_gradient,
        water_direction,
        co2_direction,
        water_direction_gradient,
        co2_direction_gradient,
        water_test_value,
        co2_test_value,
        water_test_gradient,
        co2_test_gradient,
        intrinsic_permeability,
        dt,
        temporary_prefix="two_phase_jacobian_tmp",
    ):
        args = (
            water_pressure,
            co2_pressure,
            water_pressure_old,
            co2_pressure_old,
            water_pressure_gradient,
            co2_pressure_gradient,
            water_direction,
            co2_direction,
            water_direction_gradient,
            co2_direction_gradient,
            water_test_value,
            co2_test_value,
            water_test_gradient,
            co2_test_gradient,
            intrinsic_permeability,
            dt,
        )
        expressions = self.linearized_kernel_expressions(*args)
        data_symbols = _unique_symbols(
            (
                water_pressure,
                co2_pressure,
                water_pressure_old,
                co2_pressure_old,
                water_direction,
                co2_direction,
                water_test_value,
                co2_test_value,
                dt,
            )
            + tuple(_as_vector(water_pressure_gradient, self.dim, "water_pressure_gradient"))
            + tuple(_as_vector(co2_pressure_gradient, self.dim, "co2_pressure_gradient"))
            + tuple(
                _as_vector(
                    water_direction_gradient,
                    self.dim,
                    "water_direction_gradient",
                )
            )
            + tuple(
                _as_vector(
                    co2_direction_gradient,
                    self.dim,
                    "co2_direction_gradient",
                )
            )
            + tuple(_as_vector(water_test_gradient, self.dim, "water_test_gradient"))
            + tuple(_as_vector(co2_test_gradient, self.dim, "co2_test_gradient"))
            + tuple(
                _as_square_matrix(
                    intrinsic_permeability,
                    self.dim,
                    "intrinsic_permeability",
                )
            )
            + self.constitutive.parameters.as_tuple()
        )
        return expressions.build_graph(
            data_symbols=data_symbols,
            temporary_prefix=temporary_prefix,
        )

    def build_expression_graph(
        self,
        water_pressure,
        co2_pressure,
        water_pressure_old,
        co2_pressure_old,
        water_pressure_gradient,
        co2_pressure_gradient,
        water_test_value,
        co2_test_value,
        water_test_gradient,
        co2_test_gradient,
        intrinsic_permeability,
        dt,
        temporary_prefix="two_phase_residual_tmp",
    ):
        args = (
            water_pressure,
            co2_pressure,
            water_pressure_old,
            co2_pressure_old,
            water_pressure_gradient,
            co2_pressure_gradient,
            water_test_value,
            co2_test_value,
            water_test_gradient,
            co2_test_gradient,
            intrinsic_permeability,
            dt,
        )
        expressions = self.kernel_expressions(*args)
        data_symbols = _unique_symbols(
            (
                water_pressure,
                co2_pressure,
                water_pressure_old,
                co2_pressure_old,
                dt,
            )
            + tuple(_as_vector(water_pressure_gradient, self.dim, "water_pressure_gradient"))
            + tuple(_as_vector(co2_pressure_gradient, self.dim, "co2_pressure_gradient"))
            + (
                sp.sympify(water_test_value),
                sp.sympify(co2_test_value),
            )
            + tuple(_as_vector(water_test_gradient, self.dim, "water_test_gradient"))
            + tuple(_as_vector(co2_test_gradient, self.dim, "co2_test_gradient"))
            + tuple(
                _as_square_matrix(
                    intrinsic_permeability,
                    self.dim,
                    "intrinsic_permeability",
                )
            )
            + self.constitutive.parameters.as_tuple()
        )
        return expressions.build_graph(
            data_symbols=data_symbols,
            temporary_prefix=temporary_prefix,
        )


def _as_vector(value, dim, name):
    vector = sp.Matrix(value)
    if vector.shape not in ((dim, 1), (1, dim)):
        raise ValueError("%s must contain %d entries" % (name, dim))
    return sp.Matrix(dim, 1, tuple(vector))


def _as_square_matrix(value, dim, name):
    matrix = sp.Matrix(value)
    if matrix.shape != (dim, dim):
        raise ValueError("%s must have shape (%d, %d)" % (name, dim, dim))
    return matrix


def _unique_symbols(values):
    result = []
    seen = set()
    for value in values:
        value = sp.sympify(value)
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)
