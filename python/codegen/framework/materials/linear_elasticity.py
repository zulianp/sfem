#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


mu = gen.material_parameter("mu")
lmbda = gen.material_parameter("lmbda")
element = gen.VectorElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _small_strain_energy(F, dim):
    F = F.value
    eps = (F + F.T) / 2 - sp.eye(dim)
    trace_eps = sum(eps[d, d] for d in range(dim))
    return mu * gen.inner(eps, eps) + lmbda * sp.Rational(1, 2) * trace_eps * trace_eps


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        F = gen.variable(
            gen.deformation_gradient(u),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )
        system.add_energy("", _small_strain_energy(F, dim), fields=(u,), variables=(F,))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "linear_elasticity",
    systems,
    elements=gen.sfem_supported_element_types() + ("PROTEUS_HEX125", "PROTEUS_HEX729"),
    op_name="GeneratedLinearElasticity",
    parameter_defaults=(("mu", 1.0), ("lmbda", 1.0)),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
