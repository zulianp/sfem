#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


c1 = gen.material_parameter("c1")
c2 = gen.material_parameter("c2")
kappa = gen.material_parameter("kappa")
element = gen.VectorElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _modified_mooney_rivlin_energy(F, dim):
    I1, I2, J = gen.isochoric_invariants(F, dim)

    return (
        c1 * (J ** sp.Rational(-2, 3) * I1 - 3)
        + c2 * (J ** sp.Rational(-4, 3) * I2 - 3)
        + kappa * sp.log(J) ** 2 / 2
    )


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        F = gen.variable(
            gen.Identity(dim) + gen.grad(u),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )
        system.add_energy(
            "", _modified_mooney_rivlin_energy(F, dim), fields=(u,), variables=(F,)
        )
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "modified_mooney_rivlin",
    systems,
    elements=gen.sfem_supported_element_types(),
    op_name="GeneratedModifiedMooneyRivlin",
    parameter_defaults=(("c1", 1.0), ("c2", 1.0), ("kappa", 1.0)),
    matrix_formats=("bsr",),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
