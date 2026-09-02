#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


mu = gen.material_parameter("mu")
lmbda = gen.material_parameter("lmbda")
eta_s = gen.material_parameter("eta_s")
eta_b = gen.material_parameter("eta_b")
newmark_velocity_alpha = gen.material_parameter("newmark_velocity_alpha")
element = gen.VectorElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _mooney_rivlin_energy(F, dim):
    F_value = F.value
    C = F_value.T * F_value
    J = gen.det(F_value)
    I1 = gen.inner(F_value, F_value)
    I2 = sp.Rational(1, 2) * (I1 * I1 - gen.inner(C, C))
    if dim == 2:
        # Plane-strain 3D embedding (F_zz = 1): I1 += F_zz^2, I2 from block structure of C
        I2 = I2 + I1
        I1 = I1 + 1
    # Same 3D Mooney-Rivlin form (reference invariants of Identity_3)
    return mu * (I1 - 3 + I2 - 3 - 6 * (J - 1)) + lmbda * (J - 1) ** 2 / 2


def _kelvin_voigt_residual(u, v, dim):
    grad_u = gen.grad(u)
    grad_z = gen.grad(gen.old(u))
    F = gen.Identity(dim) + grad_u
    J = gen.det(F)
    Finv = gen.inv(F)
    Fdot = newmark_velocity_alpha * grad_u + grad_z
    L = Fdot * Finv
    D = sp.Rational(1, 2) * (L + L.T)
    trD = sum(D[i, i] for i in range(dim))
    devD = D - sp.Rational(1, dim) * trD * gen.Identity(dim)
    sigma_v = 2 * eta_s * devD + eta_b * trD * gen.Identity(dim)
    P_v = J * sigma_v * Finv.T
    return gen.inner(P_v, gen.grad(v))


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        v = gen.TestFunction(V, name="u_test")
        F = gen.variable(
            gen.Identity(dim) + gen.grad(u),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )
        system.add_energy("elastic", _mooney_rivlin_energy(F, dim), fields=(u,), variables=(F,))
        system.add_residual(
            "viscous",
            _kelvin_voigt_residual(u, v, dim),
            fields=(u,),
            diagnostics=False,
        )
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "mooney_rivlin_kelvin_voigt_newmark",
    systems,
    elements=gen.sfem_supported_element_types(),
    op_name="GeneratedMooneyRivlinKelvinVoigtNewmark",
    parameter_defaults=(
        ("lmbda", 1.0),
        ("mu", 1.0),
        ("eta_s", 0.1),
        ("eta_b", 0.0),
        ("newmark_velocity_alpha", 1.0),
    ),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
