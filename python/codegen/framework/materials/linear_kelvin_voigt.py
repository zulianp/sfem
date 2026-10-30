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


def _small_strain_energy(F, dim):
    F = F.value
    eps = (F + F.T) / 2 - sp.eye(dim)
    trace_eps = sum(eps[d, d] for d in range(dim))
    return mu * gen.inner(eps, eps) + lmbda * sp.Rational(1, 2) * trace_eps * trace_eps


def _kelvin_voigt_residual(u, v, dim):
    # The small-strain counterpart of `mooney_rivlin_kelvin_voigt`'s viscous stress, with the same
    # two viscosities: sigma_v = 2 eta_s dev(eps_dot) + eta_b tr(eps_dot) I. The rate is `gen.dt`,
    # which a `TimeScheme` the operator holds fills in at run time.
    grad_u_dot = gen.grad(gen.dt(u))
    eps_dot = sp.Rational(1, 2) * (grad_u_dot + grad_u_dot.T)
    tr = sum(eps_dot[i, i] for i in range(dim))
    dev = eps_dot - sp.Rational(1, dim) * tr * gen.Identity(dim)
    sigma_v = 2 * eta_s * dev + eta_b * tr * gen.Identity(dim)
    return gen.inner(sigma_v, gen.grad(v))


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        v = gen.TestFunction(V, name="u_test")
        F = gen.variable(
            gen.deformation_gradient(u),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )
        system.add_energy("elastic", _small_strain_energy(F, dim), fields=(u,), variables=(F,))
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
    "linear_kelvin_voigt",
    systems,
    # The default set plus the level-2 semi-structured hex (with its HEX27) that the damped-mode
    # V&V case runs on, as `linear_elasticity` carries.
    elements=gen.sfem_default_element_types() + ("HEX27", "PROTEUS_HEX27"),
    op_name="GeneratedLinearKelvinVoigt",
    # A vector-valued problem uses BSR and nothing else.
    matrix_formats=("bsr",),
    parameter_defaults=(
        ("lmbda", 1.0),
        ("mu", 1.0),
        ("eta_s", 0.1),
        ("eta_b", 0.0),
        ("u_dt_shift", 1.0),
    ),
    inexact_apply=True,
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
