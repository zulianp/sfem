from dataclasses import dataclass, fields
from typing import Mapping, Tuple

import sympy as sp

from .symbolic import ExpressionRole, KernelExpressions


def _dataclass_value_tuple(instance):
    return tuple(getattr(instance, field.name) for field in fields(instance))


def _dataclass_value_dict(instance):
    return {field.name: getattr(instance, field.name) for field in fields(instance)}


@dataclass(frozen=True)
class TwoPhaseFlowParameters:
    porosity: sp.Symbol
    residual_water_saturation: sp.Symbol
    reference_capillary_pressure: sp.Symbol
    retention_exponent: sp.Symbol
    reference_water_density: sp.Symbol
    water_compressibility: sp.Symbol
    reference_water_pressure: sp.Symbol
    co2_molar_mass: sp.Symbol
    co2_compressibility_factor: sp.Symbol
    gas_constant: sp.Symbol
    temperature: sp.Symbol
    water_viscosity: sp.Symbol
    co2_viscosity: sp.Symbol
    water_permeability_exponent: sp.Symbol
    co2_permeability_exponent_1: sp.Symbol
    co2_permeability_exponent_2: sp.Symbol

    @classmethod
    def symbols(cls, prefix=""):
        names = (
            "porosity",
            "S_res",
            "P_r",
            "m",
            "rho_w0",
            "kappa_T",
            "p_wr",
            "M_c",
            "Z",
            "R",
            "T",
            "mu_w",
            "mu_c",
            "C_kw1",
            "C_ka1",
            "C_ka2",
        )
        return cls(*sp.symbols(" ".join("%s%s" % (prefix, name) for name in names)))

    def as_tuple(self) -> Tuple[sp.Symbol, ...]:
        return _dataclass_value_tuple(self)

    def as_dict(self):
        return _dataclass_value_dict(self)

    def validate(self, values: Mapping):
        resolved = {
            field.name: _parameter_value(values, field.name, getattr(self, field.name))
            for field in fields(self)
        }
        if not 0.0 < resolved["porosity"] <= 1.0:
            raise ValueError("porosity must be in (0, 1]")
        if not 0.0 <= resolved["residual_water_saturation"] < 1.0:
            raise ValueError("residual_water_saturation must be in [0, 1)")
        positive = (
            "reference_capillary_pressure",
            "reference_water_density",
            "co2_molar_mass",
            "co2_compressibility_factor",
            "gas_constant",
            "temperature",
            "water_viscosity",
            "co2_viscosity",
            "water_permeability_exponent",
            "co2_permeability_exponent_1",
            "co2_permeability_exponent_2",
        )
        for name in positive:
            if resolved[name] <= 0.0:
                raise ValueError("%s must be positive" % name)
        if resolved["retention_exponent"] < 1.0:
            raise ValueError(
                "retention_exponent must be at least 1 for an admissible saturation"
            )
        if resolved["water_compressibility"] < 0.0:
            raise ValueError("water_compressibility must be nonnegative")
        return resolved


@dataclass(frozen=True)
class TwoPhaseFlowConstitutiveState:
    capillary_pressure: sp.Expr
    water_saturation: sp.Expr
    co2_saturation: sp.Expr
    effective_water_saturation: sp.Expr
    water_density: sp.Expr
    co2_density: sp.Expr
    water_relative_permeability: sp.Expr
    co2_relative_permeability: sp.Expr
    water_mobility: sp.Expr
    co2_mobility: sp.Expr

    def as_tuple(self):
        return _dataclass_value_tuple(self)

    def as_dict(self):
        return _dataclass_value_dict(self)


@dataclass(frozen=True)
class TwoPhaseFlowConstitutiveModel:
    parameters: TwoPhaseFlowParameters

    @classmethod
    def symbolic(cls, prefix=""):
        return cls(TwoPhaseFlowParameters.symbols(prefix))

    def state(self, water_pressure, co2_pressure):
        p = self.parameters
        suction = co2_pressure - water_pressure
        water_saturation = p.residual_water_saturation + (
            1 - p.residual_water_saturation
        ) * (
            1 + (suction / p.reference_capillary_pressure) ** p.retention_exponent
        ) ** (
            1 / p.retention_exponent - 1
        )
        co2_saturation = 1 - water_saturation
        effective_water_saturation = (
            water_saturation - p.residual_water_saturation
        ) / (
            1 - p.residual_water_saturation
        )
        water_density = p.reference_water_density * sp.exp(
            p.water_compressibility
            * (water_pressure - p.reference_water_pressure)
        )
        co2_density = (
            p.co2_molar_mass
            * co2_pressure
            / (
                p.co2_compressibility_factor
                * p.gas_constant
                * p.temperature
            )
        )
        water_relative_permeability = sp.sqrt(water_saturation) * (
            1
            - (
                1
                - water_saturation
                ** (1 / p.water_permeability_exponent)
            )
            ** p.water_permeability_exponent
        ) ** 2
        co2_relative_permeability = (
            1 - effective_water_saturation
        ) ** p.co2_permeability_exponent_1 * (
            1
            - effective_water_saturation
            ** p.co2_permeability_exponent_2
        )
        return TwoPhaseFlowConstitutiveState(
            capillary_pressure=suction,
            water_saturation=water_saturation,
            co2_saturation=co2_saturation,
            effective_water_saturation=effective_water_saturation,
            water_density=water_density,
            co2_density=co2_density,
            water_relative_permeability=water_relative_permeability,
            co2_relative_permeability=co2_relative_permeability,
            water_mobility=water_relative_permeability / p.water_viscosity,
            co2_mobility=co2_relative_permeability / p.co2_viscosity,
        )

    def pressure_derivatives(self, water_pressure, co2_pressure):
        state = self.state(water_pressure, co2_pressure)
        return {
            name: (
                sp.diff(expression, water_pressure),
                sp.diff(expression, co2_pressure),
            )
            for name, expression in state.as_dict().items()
        }

    def kernel_expressions(
        self,
        water_pressure,
        co2_pressure,
        include_derivatives=False,
    ):
        state = self.state(water_pressure, co2_pressure)
        expressions = KernelExpressions()
        for name, expression in state.as_dict().items():
            expressions.add(ExpressionRole.OPERATOR_EVALUATION, expression, name)
        if include_derivatives:
            for name, derivatives in self.pressure_derivatives(
                water_pressure,
                co2_pressure,
            ).items():
                expressions.add(
                    ExpressionRole.OPERATOR_EVALUATION,
                    derivatives[0],
                    "d_%s_d_pw" % name,
                )
                expressions.add(
                    ExpressionRole.OPERATOR_EVALUATION,
                    derivatives[1],
                    "d_%s_d_pc" % name,
                )
        return expressions

    def build_expression_graph(
        self,
        water_pressure,
        co2_pressure,
        include_derivatives=False,
        temporary_prefix="two_phase_tmp",
    ):
        return self.kernel_expressions(
            water_pressure,
            co2_pressure,
            include_derivatives,
        ).build_graph(
            data_symbols=(
                water_pressure,
                co2_pressure,
            )
            + self.parameters.as_tuple(),
            temporary_prefix=temporary_prefix,
        )

    def validate_state(self, water_pressure, co2_pressure):
        if co2_pressure <= 0.0:
            raise ValueError("co2_pressure must be positive")
        if co2_pressure < water_pressure:
            raise ValueError("capillary pressure p_c - p_w must be nonnegative")


def _parameter_value(values, name, symbol):
    if symbol in values:
        return float(values[symbol])
    if name in values:
        return float(values[name])
    if str(symbol) in values:
        return float(values[str(symbol)])
    raise ValueError("missing two-phase parameter '%s'" % name)
