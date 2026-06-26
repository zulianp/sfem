#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


mu, lmbda, alpha, storage, dt, hydraulic_conductivity = sp.symbols(
    "mu lmbda alpha storage dt hydraulic_conductivity"
)


def strain_energy(F):
    dim = F.rows
    log_j = sp.log(F.det())
    return (
        mu * (gen.matrix_inner(F, F) - dim) / 2
        - mu * log_j
        + lmbda * log_j**2 / 2
    )


def pressure_residual(system):
    displacement = tuple(
        system.add_field("u%d" % d)
        for d in range(system.dim)
    )
    pressure = system.add_field("p")
    system.add_parameters(alpha, storage, dt, hydraulic_conductivity)

    div_u = sum(displacement[d].gradient[d] for d in range(system.dim))
    div_u_old = sum(displacement[d].previous_gradient[d] for d in range(system.dim))

    for d, field in enumerate(displacement):
        system.set_residual(
            field,
            -alpha * pressure.value * field.test_gradient[d],
        )

    diffusion = hydraulic_conductivity * sum(
        pressure.gradient[d] * pressure.test_gradient[d]
        for d in range(system.dim)
    )
    accumulation = (
        storage * (pressure.value - pressure.previous_value)
        + alpha * (div_u - div_u_old)
    ) * pressure.test_value / dt
    system.set_residual(pressure, accumulation + diffusion)


def define(system):
    displacement = system.vector_field("u", family="displacement")
    pressure = system.scalar_field("p", family="pressure")
    system.hyperelastic("solid", strain_energy, fields=(displacement,))
    system.residual("poro", pressure_residual, fields=(displacement, pressure))


material = gen.UnifiedMaterial(
    "poro_hyperelasticity",
    define,
    elements=gen.sfem_taylor_hood_element_types(),
    parameter_defaults=(
        ("mu", 1.0),
        ("lmbda", 1.0),
        ("alpha", 0.8),
        ("storage", 1.0e-3),
        ("dt", 1.0),
        ("hydraulic_conductivity", 1.0),
    ),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
