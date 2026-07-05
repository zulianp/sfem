#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


element = gen.VectorElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        v = gen.TestFunction(V, name="u_test")
        traction = sp.Matrix([gen.material_parameter("t%d" % d) for d in range(dim)])
        system.add_residual("", -gen.inner(traction, v) * gen.ds, fields=(u,))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "neumann",
    systems,
    op_name="GeneratedNeumann",
    elements=(
        "TRI3",
        "QUAD4",
        "TET4",
        "TET10",
        "HEX8",
        "HEX27",
        "PROTEUS_HEX8",
        "PROTEUS_HEX27",
        "PROTEUS_HEX64",
        "PROTEUS_HEX125",
    ),
    parameter_defaults=(("t0", 0.0), ("t1", 0.0), ("t2", 0.0)),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
