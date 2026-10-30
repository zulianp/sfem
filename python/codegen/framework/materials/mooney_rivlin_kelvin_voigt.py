#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


mu = gen.material_parameter("mu")
lmbda = gen.material_parameter("lmbda")
eta_s = gen.material_parameter("eta_s")
eta_b = gen.material_parameter("eta_b")
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
    F = gen.Identity(dim) + grad_u
    J = gen.det(F)
    Finv = gen.inv(F)
    # The strain rate, and no integration scheme.  `gen.dt` carries the time
    # derivative as a weighted sum a `TimeScheme` fills in at run time, so this
    # form is the Kelvin-Voigt material and nothing about how it is stepped.
    # It also has to reach here rather than ride an additive inertia operator:
    # `Fdot` enters through `Finv` and `J`, so the rate is inside the
    # constitutive law, not beside it.
    Fdot = gen.grad(gen.dt(u))
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
    "mooney_rivlin_kelvin_voigt",
    systems,
    elements=gen.sfem_default_element_types(),
    op_name="GeneratedMooneyRivlinKelvinVoigt",
    # Both units assemble: the elastic energy through the direct element-matrix
    # kernel and the viscous residual through its own.  A vector-valued problem
    # uses BSR and nothing else.
    matrix_formats=("bsr",),
    parameter_defaults=(
        ("lmbda", 1.0),
        ("mu", 1.0),
        ("eta_s", 0.1),
        ("eta_b", 0.0),
        ("u_dt_shift", 1.0),
    ),
    # The stored-tangent apply: the total residual's tangent at every point, 81 numbers (the
    # viscous term reads F^-1, so the Jacobian is unsymmetric) and the previous state as data.
    # It costs about 11 s per element on top of the material's own symbolic set-up.
    inexact_apply=True,
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
