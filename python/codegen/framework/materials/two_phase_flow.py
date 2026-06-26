#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


def define(system):
    water = system.Coefficient("p_w")
    co2 = system.Coefficient("p_c")
    q_w = gen.TestFunction(water)
    q_c = gen.TestFunction(co2)

    constitutive = gen.TwoPhaseFlowConstitutiveModel.symbolic()
    dt = sp.Symbol("dt")
    permeability = sp.Matrix(
        system.dim,
        system.dim,
        sp.symbols("K_0:%d" % (system.dim * system.dim)),
    )

    current = constitutive.state(water, co2)
    previous = constitutive.state(gen.old(water), gen.old(co2))
    porosity = constitutive.parameters.porosity
    water_accumulation = (
        porosity
        * (
            current.water_saturation * current.water_density
            - previous.water_saturation * previous.water_density
        )
        / dt
    )
    co2_accumulation = (
        porosity
        * (
            current.co2_saturation * current.co2_density
            - previous.co2_saturation * previous.co2_density
        )
        / dt
    )
    water_flux = -(
        current.water_density
        * current.water_mobility
        * permeability
        * gen.grad(water)
    )
    co2_flux = -(
        current.co2_density
        * current.co2_mobility
        * permeability
        * gen.grad(co2)
    )

    form = (
        water_accumulation * q_w
        - water_flux.dot(gen.grad(q_w))
        + co2_accumulation * q_c
        - co2_flux.dot(gen.grad(q_c))
    )
    system.residual("", form, fields=(water, co2))


material = gen.UnifiedMaterial(
    "two_phase_flow",
    define,
    elements=("TRI3", "TET4", "QUAD4", "HEX8"),
    op_name="GeneratedTwoPhaseFlow",
    parameter_defaults=(
        ("porosity", 0.1),
        ("S_res", 0.39),
        ("P_r", 9.5e4 / 1.0e6),
        ("m", 4.2),
        ("rho_w0", 1100.0),
        ("kappa_T", 4.55e-10 * 1.0e6),
        ("p_wr", 1.0e6 / 1.0e6),
        ("M_c", 0.04401),
        ("Z", 0.4252),
        ("R", 8.314 / 1.0e6),
        ("T", 333.0),
        ("mu_w", 5.2),
        ("mu_c", 1.5),
        ("C_kw1", 0.52),
        ("C_ka1", 1.8),
        ("C_ka2", 0.35),
        ("dt", 1.0),
        ("K_0", 86.40),
        ("K_1", 0.0),
        ("K_2", 0.0),
        ("K_3", 0.0),
        ("K_4", 86.40),
        ("K_5", 0.0),
        ("K_6", 0.0),
        ("K_7", 0.0),
        ("K_8", 86.40),
    ),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
