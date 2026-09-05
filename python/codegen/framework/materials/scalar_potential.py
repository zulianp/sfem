#!/usr/bin/env python3
"""A scalar field's energy: the smallest material that is not a displacement.

Every energy material the framework had was a displacement, whose field has as
many components as the domain has dimensions.  Nothing distinguished the two
numbers, so the emitters used `dim` for both and a scalar field generated
kernels that read one component through an ABI declaring three, indexed a
four-entry array with a stride of three, and answered zero.  Each of those was
found by hand, one at a time, because no maintained material exercised the
path.

This is that material.  Its energy `kappa/2 * ||grad u||^2` is the potential
whose gradient is the Laplacian residual, so `laplace` is its reference: the
same operator declared the other way round, already generated and already
gated.  Where the two disagree, one of them is wrong.
"""

from pathlib import Path

from sfem import gen


kappa = gen.material_parameter("kappa")
element = gen.FiniteElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u")
        # One field component, `dim` directions -- a row, not a column.
        G = gen.variable(gen.grad(u).T, name="G")
        system.add_energy(
            "", kappa / 2 * gen.inner(G, G), fields=(u,), variables=(G,)
        )
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "scalar_potential",
    systems,
    elements=("TRI3", "QUAD4", "TET4", "HEX8"),
    op_name="GeneratedScalarPotential",
    parameter_defaults=(("kappa", 1.0),),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
