#!/usr/bin/env python3
from pathlib import Path

from sfem import gen


density = gen.material_parameter("density")
element = gen.VectorElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        v = gen.TestFunction(V, name="u_test")
        # The consistent mass, rho u . v: its action is M u and its 2-form the mass matrix itself,
        # on every element the elastic materials carry. A time scheme's inertia uses it in place of
        # the lumped mass when a case asks for the consistent one, and forms the potential
        # 1/2 u^T M u from the action. (A residual rather than an energy because the energy form is
        # written in the deformation gradient, and this one has none.)
        system.add_residual("", density * gen.inner(u, v), fields=(u,))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "vector_mass",
    systems,
    # The elements of the materials it runs beside, with the level-2 semi-structured hex.
    elements=gen.sfem_default_element_types() + ("HEX27", "PROTEUS_HEX27"),
    op_name="GeneratedVectorMass",
    parameter_defaults=(("density", 1.0),),
    matrix_formats=("bsr",),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
