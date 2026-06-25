#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


def weak_form(system):
    water = system.add_field("p_w")
    co2 = system.add_field("p_c")
    constitutive = gen.TwoPhaseFlowConstitutiveModel.symbolic()
    dt = sp.Symbol("dt")
    permeability = sp.Matrix(
        system.dim,
        system.dim,
        sp.symbols("K_0:%d" % (system.dim * system.dim)),
    )
    system.add_parameters(
        *constitutive.parameters.as_tuple(),
        dt,
        *tuple(permeability),
    )

    current = constitutive.state(water.value, co2.value)
    previous = constitutive.state(water.previous_value, co2.previous_value)
    porosity = constitutive.parameters.porosity
    water_accumulation = porosity * (
        current.water_saturation * current.water_density
        - previous.water_saturation * previous.water_density
    ) / dt
    co2_accumulation = porosity * (
        current.co2_saturation * current.co2_density
        - previous.co2_saturation * previous.co2_density
    ) / dt
    water_flux = -(
        current.water_density
        * current.water_mobility
        * permeability
        * sp.Matrix(water.gradient)
    )
    co2_flux = -(
        current.co2_density
        * current.co2_mobility
        * permeability
        * sp.Matrix(co2.gradient)
    )

    system.set_residual(
        water,
        water_accumulation * water.test_value
        - water_flux.dot(sp.Matrix(water.test_gradient)),
    )
    system.set_residual(
        co2,
        co2_accumulation * co2.test_value
        - co2_flux.dot(sp.Matrix(co2.test_gradient)),
    )


material = gen.CoupledResidualMaterial(
    "two_phase_flow",
    weak_form,
    elements=("TRI3", "TET4", "QUAD4", "HEX8"),
    op_name="GeneratedTwoPhaseFlow",
    parameter_defaults=(
        ("porosity", 0.2),
        ("S_res", 0.1),
        ("P_r", 1.0e5),
        ("m", 2.0),
        ("rho_w0", 1000.0),
        ("kappa_T", 1.0e-9),
        ("p_wr", 1.0e5),
        ("M_c", 0.044),
        ("Z", 1.0),
        ("R", 8.314462618),
        ("T", 300.0),
        ("mu_w", 1.0e-3),
        ("mu_c", 1.5e-5),
        ("C_kw1", 2.0),
        ("C_ka1", 2.0),
        ("C_ka2", 2.0),
        ("dt", 1.0),
        ("K_0", 1.0e-12),
        ("K_1", 0.0),
        ("K_2", 0.0),
        ("K_3", 0.0),
        ("K_4", 1.0e-12),
        ("K_5", 0.0),
        ("K_6", 0.0),
        ("K_7", 0.0),
        ("K_8", 1.0e-12),
    ),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
