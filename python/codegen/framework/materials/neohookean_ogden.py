#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


mu = gen.material_parameter("mu")
lmbda = gen.material_parameter("lmbda")


def define(system):
    u = system.VectorCoefficient("u", family="displacement")

    F = gen.variable(
        gen.Identity(system.dim) + gen.grad(u),
        name="F",
        qualifier=gen.DEFORMATION_GRADIENT,
    )
    J = gen.det(F)
    psi = (
        mu * (gen.inner(F, F) - system.dim) / 2
        - mu * sp.log(J)
        + lmbda * sp.log(J) ** 2 / 2
    )

    system.energy("", psi, fields=(u,))


material = gen.UnifiedMaterial(
    "neohookean_ogden",
    define,
    elements=gen.sfem_supported_element_types(),
    op_name="GeneratedNeoHookeanOgden",
    parameter_defaults=(("mu", 1.0), ("lmbda", 1.0)),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
