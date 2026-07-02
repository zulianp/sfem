#!/usr/bin/env python3
from pathlib import Path

from sfem import gen


kappa = gen.material_parameter("kappa")
element = gen.FiniteElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u")
        v = gen.TestFunction(V, name="u_test")
        system.add_residual("", kappa * gen.inner(gen.grad(u), gen.grad(v)), fields=(u,))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "laplace",
    systems,
    elements=gen.sfem_supported_element_types(),
    op_name="GeneratedLaplace",
    parameter_defaults=(("kappa", 1.0),),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
