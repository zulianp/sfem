#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


mu = gen.material_parameter("mu")
lmbda = gen.material_parameter("lmbda")
alpha = gen.material_parameter("alpha")
storage = gen.material_parameter("storage")
dt = gen.material_parameter("dt")
hydraulic_conductivity = gen.material_parameter("hydraulic_conductivity")


def define(system):
    u = system.VectorCoefficient("u", family="displacement")
    p = system.Coefficient("p", family="pressure")
    v = gen.TestFunction(u)
    q = gen.TestFunction(p)

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
    system.energy("solid", psi, fields=(u,))

    form = (
        -alpha * p * gen.div(v)
        + (
            storage * (p - gen.old(p))
            + alpha * (gen.div(u) - gen.div(gen.old(u)))
        )
        * q
        / dt
        + hydraulic_conductivity * gen.inner(gen.grad(p), gen.grad(q))
    )
    system.residual("poro", form, fields=(u, p))


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
