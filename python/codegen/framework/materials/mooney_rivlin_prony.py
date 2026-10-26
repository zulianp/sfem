#!/usr/bin/env python3
"""Mooney-Rivlin with a Prony-series relaxation of its isochoric part.

The law of SFEM's hand-written `MooneyRivlinVisco`, which stays as the
reference this one is measured against:

    psi = C10 (I1bar - 3) + C01 (I2bar - 3) + K/2 (J - 1)^2,

with I1bar = J^(-2/3) I1 and I2bar = J^(-4/3) I2.  The isochoric part relaxes
and the volumetric part does not.  Nothing here is particular to Mooney-Rivlin:
any energy split into an equilibrium and a relaxed part is extended the same
way, by handing the relaxed part to `PronyRelaxation.relax`.
"""

from pathlib import Path

import sympy as sp

from sfem import gen


C10 = gen.material_parameter("C10")
C01 = gen.material_parameter("C01")
K = gen.material_parameter("K")
element = gen.VectorElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def isochoric_energy(F, dim):
    I1, I2, J = gen.isochoric_invariants(F, dim)
    return C10 * (J ** sp.Rational(-2, 3) * I1 - 3) + C01 * (J ** sp.Rational(-4, 3) * I2 - 3)


def volumetric_energy(F, dim):
    J = gen.det(F.value)
    return K * (J - 1) ** 2 / 2


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        F = gen.variable(
            gen.Identity(dim) + gen.grad(u),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )
        iso = gen.PronyRelaxation("iso")
        psi = volumetric_energy(F, dim) + iso.relax(isochoric_energy(F, dim), F)
        system.add_energy("", psi, fields=(u,), variables=(F,), relaxations=(iso,))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "mooney_rivlin_prony",
    systems,
    elements=gen.sfem_default_element_types(),
    # The history lives at one rule's points, and TET10's affine and
    # isoparametric routes integrate with different rules: pinned to the
    # isoparametric one, the more accurate.
    quadrature_settings=(("TET10", 4),),
    op_name="GeneratedMooneyRivlinProny",
    parameter_defaults=(
        ("C01", 0.0),
        ("C10", 1.0),
        ("K", 1.0),
    ),
    matrix_formats=("bsr",),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
