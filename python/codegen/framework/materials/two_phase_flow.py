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
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
