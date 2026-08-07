#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


rho = gen.material_parameter("rho")
nu = gen.material_parameter("nu")
dt = gen.material_parameter("dt")
convection_scale = gen.material_parameter("convection_scale")
body_force = tuple(gen.material_parameter("f%d" % d) for d in range(3))

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


def _convective_acceleration(velocity, advecting_velocity, dim):
    gradient = gen.grad(velocity)
    return sp.Matrix(
        dim,
        1,
        lambda row, _: sum(advecting_velocity[col] * gradient[row, col] for col in range(dim)),
    )


def _body_force(dim):
    return sp.Matrix(dim, 1, lambda row, _: body_force[row])


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        velocity = gen.Function(W[0], "u", qualifier=gen.VELOCITY)
        pressure = gen.Function(W[1], "p", qualifier=gen.PRESSURE)
        test_velocity = gen.TestFunction(W[0], name="u_test")
        test_pressure = gen.TestFunction(W[1], name="p_test")

        velocity_vector = sp.Matrix(dim, 1, lambda row, _: velocity[row])
        previous_velocity = gen.old(velocity)
        previous_velocity_vector = sp.Matrix(dim, 1, lambda row, _: previous_velocity[row])
        acceleration = (velocity_vector - previous_velocity_vector) / dt
        convection = _convective_acceleration(velocity, previous_velocity, dim)

        form = (
            rho * gen.inner(acceleration, test_velocity)
            + rho * convection_scale * gen.inner(convection, test_velocity)
            + 2 * rho * nu * gen.inner(_symgrad(velocity), _symgrad(test_velocity))
            - pressure * gen.div(test_velocity)
            + test_pressure * gen.div(velocity)
            - rho * gen.inner(_body_force(dim), test_velocity)
        )
        system.add_residual("", form, fields=(velocity, pressure))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "navier_stokes",
    systems,
    op_name="GeneratedNavierStokes",
    parameter_defaults=(
        ("rho", 1.0),
        ("nu", 1.0e-3),
        ("dt", 1.0),
        ("convection_scale", 1.0),
        ("f0", 0.0),
        ("f1", 0.0),
        ("f2", 0.0),
    ),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
