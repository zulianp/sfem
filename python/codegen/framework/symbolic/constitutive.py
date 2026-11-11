from dataclasses import dataclass, fields
from typing import Mapping, Tuple

import sympy as sp

from codegen.framework.symbolic import ExpressionRole, KernelExpressions
from codegen.framework.symbolic.operators import det, inner


def isochoric_invariants(F, dim):
    """(I1, I2, J) of the deformation gradient F, embedded in 3D.

    The first two invariants of C = F^T F, unmodified -- a material divides by
    J^(2/3) and J^(4/3) itself, so the isochoric and the volumetric part of an
    energy stay separately readable.  In 2D the problem is plane strain,
    F_zz = 1, and I1 and I2 are those of the 3D embedding, so a 3D law means
    the same thing in both.
    """
    F_value = getattr(F, "value", F)
    C = F_value.T * F_value
    J = det(F_value)
    I1 = inner(F_value, F_value)
    I2 = sp.Rational(1, 2) * (I1 * I1 - inner(C, C))
    if dim == 2:
        # Plane-strain 3D embedding (F_zz = 1): I1 += F_zz^2, I2 from block structure of C
        I2 = I2 + I1
        I1 = I1 + 1
    return I1, I2, J


def continued_power_piecewise(J, m, Jc, Jmin):
    """h_m(J) written out: the definition the kernel math header implements.

    J^(-m) for J >= Jc.  For Jmin < J < Jc the quartic in z = (J - Jc) / (Jc - Jmin)
    whose value, slope and curvature match J^(-m) at Jc (a0, a1, a2) and whose
    slope and curvature vanish at z = -1 (a3, a4).  Constant for J <= Jmin.
    """
    delta = Jc - Jmin
    hc = Jc ** -m
    a1 = -m * delta * hc / Jc
    a2 = m * (m + 1) * delta**2 * hc / (2 * Jc**2)
    a3 = sp.Rational(4, 3) * a2 - a1
    a4 = (a2 - a1) / 2
    z = (J - Jc) / delta
    quartic = hc + a1 * z + a2 * z**2 + a3 * z**3 + a4 * z**4
    return sp.Piecewise((J**-m, J >= Jc), (quartic, J > Jmin), (quartic.subs(J, Jmin), True))


class _ContinuedPowerDerivative(sp.Function):
    """d^k h_m/dJ^k with arguments (J, m, Jc, Jmin); k is the class's `order`.

    Kept opaque so the generated expressions call the header's function rather
    than carrying the pieces; differentiated through `fdiff` and evaluated
    numerically from `continued_power_piecewise`, which the generator's numeric
    checks (the tangent's symmetry) need.
    """

    nargs = 4
    order = 0
    derivative = None

    def fdiff(self, argindex=1):
        if argindex != 1 or self.derivative is None:
            raise sp.ArgumentIndexError(self, argindex)
        return self.derivative(*self.args)

    def _eval_evalf(self, prec):
        if not all(argument.is_number for argument in self.args):
            return None
        J, m, Jc, Jmin = self.args
        x = sp.Dummy("J")
        value = sp.diff(continued_power_piecewise(x, m, Jc, Jmin), x, self.order).subs(x, J)
        return value.evalf(prec)


class sfem_continued_power_d2(_ContinuedPowerDerivative):
    """d2h_m/dJ2; the header has no third derivative."""

    order = 2


class sfem_continued_power_d1(_ContinuedPowerDerivative):
    """dh_m/dJ."""

    order = 1
    derivative = sfem_continued_power_d2


class sfem_continued_power(_ContinuedPowerDerivative):
    """h_m(J): J^(-m) for J >= Jc, continued below Jc so that it stays finite.

    An energy written in h_m(J) instead of J^(-m) is C2 through J = 0, so a
    Newton trial may pass through an inverted element.  The definition is
    `continued_power_piecewise`; the kernel math header implements it
    (`emitters/cprinter.py`).
    """

    order = 0
    derivative = sfem_continued_power_d1


#: The kernel math header's function for each, by the name the printer prints.
CONTINUED_POWER_FUNCTIONS = (
    "sfem_continued_power",
    "sfem_continued_power_d1",
    "sfem_continued_power_d2",
)


def continued_power(J, m, Jc, Jmin):
    """h_m(J), the continuation of J^(-m) below Jc (`sfem_continued_power`)."""
    return sfem_continued_power(J, sp.sympify(m), Jc, Jmin)


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
