#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


mu = gen.material_parameter("mu")
lmbda = gen.material_parameter("lmbda")
alpha = gen.material_parameter("alpha")
storage = gen.material_parameter("storage")
dt = gen.material_parameter("dt")
hydraulic_conductivity = gen.material_parameter("hydraulic_conductivity")
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


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(W[0], "u", qualifier=gen.DISPLACEMENT)
        p = gen.Function(W[1], "p", qualifier=gen.PRESSURE)
        v = gen.TestFunction(W[0], name="u_test")
        q = gen.TestFunction(W[1], name="p_test")

        F = gen.variable(
            gen.Identity(system.dim) + gen.grad(u),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )
        J = gen.det(F)
        psi = (
            mu * (gen.inner(F, F) - system.dim) / 2
            - mu * sp.log(J)
            + lmbda * sp.log(J) ** 2 / 2
        )
        system.add_energy("solid", psi, fields=(u,), variables=(F,))

        form = (
            -alpha * p * gen.div(v)
            + (
                storage * (p - gen.old(p))
                + alpha * (gen.div(u) - gen.div(gen.old(u)))
            )
            * q
            / dt
            + hydraulic_conductivity * gen.inner(gen.grad(p), gen.grad(q))
        )
        system.add_residual("poro", form, fields=(u, p))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "poro_hyperelasticity",
    systems,
    elements=gen.sfem_taylor_hood_element_types(),
    parameter_defaults=(
        ("mu", 1.0),
        ("lmbda", 1.0),
        ("alpha", 0.8),
        ("storage", 1.0e-3),
        ("dt", 1.0),
        ("hydraulic_conductivity", 1.0),
    ),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
