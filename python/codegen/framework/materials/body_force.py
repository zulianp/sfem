#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


density = gen.material_parameter("density")
element = gen.VectorElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _force(dim):
    return sp.Matrix([gen.material_parameter("g%d" % d) for d in range(dim)])


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        # The potential, not the residual.  A body force is the linear work
        # `-rho * g . u`, and declaring it that way is what makes this material
        # nine lines: the 0-form is the work itself, the 1-form is the load
        # vector `-rho * g . v`, and the 2-form is identically zero because the
        # work is linear in `u`.  `laplace.py` states the same reason for the
        # Dirichlet energy -- one statement, all three orders.
        #
        # Declaring the residual instead would give the same 1-form and a
        # 0-form the volume wrapper reads as a MERIT rather than a POTENTIAL,
        # which would promote every `Function` holding a body force to the
        # node-wise merit.  Gravity is a potential; saying so keeps an energy
        # line search working.
        #
        # The sign is the one every forcing operator in the tree uses:
        # `gradient` assembles `-F_ext`, matching `neumann.py` and the
        # `- rho * inner(f, v)` term `navier_stokes.py` already carries.
        v = gen.TestFunction(V, name="u_test")
        system.add_residual(
            "", -density * gen.inner(_force(dim), v), fields=(u,)
        )
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "body_force",
    systems,
    op_name="GeneratedBodyForce",
    elements=gen.sfem_default_element_types(),
    # Three components in both dimensions, so the parameter count -- and with
    # it the `Parameters` layout -- does not depend on the dimension.  The same
    # reason `neumann.py` declares `t0`, `t1`, `t2` for a 2D traction.
    parameter_defaults=(("density", 1.0), ("g0", 0.0), ("g1", 0.0), ("g2", 0.0)),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
