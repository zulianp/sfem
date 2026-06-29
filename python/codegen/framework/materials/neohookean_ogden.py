#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


mu = gen.material_parameter("mu")
lmbda = gen.material_parameter("lmbda")
element = gen.VectorElement(
    "Lagrange",
    degree=1,
)
V = gen.FunctionSpace(element)


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)

        F = gen.variable(
            gen.Identity(dim) + gen.grad(u),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )
        J = gen.det(F)
        psi = (
            mu * (gen.inner(F, F) - dim) / 2
            - mu * sp.log(J)
            + lmbda * sp.log(J) ** 2 / 2
        )

        system.add_energy("", psi, fields=(u,), variables=(F,))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "neohookean_ogden",
    systems,
    elements=gen.sfem_supported_element_types() + ("PROTEUS_HEX125", "PROTEUS_HEX729"),
    op_name="GeneratedNeoHookeanOgden",
    parameter_defaults=(("mu", 1.0), ("lmbda", 1.0)),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
