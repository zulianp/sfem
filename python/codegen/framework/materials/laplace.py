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
        # The Dirichlet energy, whose gradient is the residual this used to
        # declare.  Written as the potential so that `value` and `value_steps`
        # answer with an energy rather than a merit, and so the 0-form, the
        # gradient and the Hessian action all come from one statement.
        G = gen.variable(gen.grad(u).T, name="G")
        system.add_energy(
            "", kappa / 2 * gen.inner(G, G), fields=(u,), variables=(G,)
        )
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "laplace",
    systems,
    elements=gen.sfem_supported_element_types() + ("PROTEUS_HEX125", "PROTEUS_HEX729"),
    op_name="GeneratedLaplace",
    parameter_defaults=(("kappa", 1.0),),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
