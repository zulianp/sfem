#!/usr/bin/env python3
from pathlib import Path

from sfem import gen


mu = gen.material_parameter("mu")
V = gen.FunctionSpace(
    gen.VectorElement(
        "Lagrange",
        degree=2,
    )
)
Q = gen.FunctionSpace(
    gen.FiniteElement(
        "Lagrange",
        degree=1,
    )
)
W = gen.MixedFunctionSpace(V, Q)


def _symgrad(field):
    gradient = gen.grad(field)
    return (gradient + gradient.T) / 2


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        velocity = gen.Function(W[0], "u", qualifier=gen.VELOCITY)
        pressure = gen.Function(W[1], "p", qualifier=gen.PRESSURE)
        test_velocity = gen.TestFunction(W[0], name="u_test")
        test_pressure = gen.TestFunction(W[1], name="p_test")

        form = (
            2 * mu * gen.inner(_symgrad(velocity), _symgrad(test_velocity))
            - pressure * gen.div(test_velocity)
            + test_pressure * gen.div(velocity)
        )
        system.add_residual("", form, fields=(velocity, pressure))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "stokes",
    systems,
    op_name="GeneratedStokes",
    parameter_defaults=(("mu", 1.0),),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
