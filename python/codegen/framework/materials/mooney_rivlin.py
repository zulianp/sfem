#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


mu = gen.material_parameter("mu")
lmbda = gen.material_parameter("lmbda")
element = gen.VectorElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _mooney_rivlin_energy(F, dim):
    F_value = F.value
    C = F_value.T * F_value
    J = gen.det(F_value)
    I1 = gen.inner(F_value, F_value)
    I2 = sp.Rational(1, 2) * (I1 * I1 - gen.inner(C, C))
    I2_reference = sp.Rational(dim * (dim - 1), 2)
    return mu * (I1 - dim + I2 - I2_reference) + lmbda * (J - 1) ** 2 / 2


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        F = gen.variable(
            gen.Identity(dim) + gen.grad(u),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )
        system.add_energy("", _mooney_rivlin_energy(F, dim), fields=(u,), variables=(F,))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "mooney_rivlin",
    systems,
    elements=gen.sfem_supported_element_types(),
    op_name="GeneratedMooneyRivlin",
    parameter_defaults=(("mu", 1.0), ("lmbda", 1.0)),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
