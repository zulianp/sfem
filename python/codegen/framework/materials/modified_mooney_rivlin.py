#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


c1 = gen.material_parameter("c1")
c2 = gen.material_parameter("c2")
kappa = gen.material_parameter("kappa")
element = gen.VectorElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _modified_mooney_rivlin_energy(F, dim):
    F_value = F.value
    C = F_value.T * F_value
    J = gen.det(F_value)
    I1 = gen.inner(F_value, F_value)
    I2 = sp.Rational(1, 2) * (I1 * I1 - gen.inner(C, C))

    if dim == 2:
        # Plane-strain embedding: F is 2×2 but represents a 3D body with F_33=1.
        # J is unchanged (det(F_3D) = det(F_2D)·1 = J).
        # I1_3D = I1_2D + 1  (F_33=1 contributes F_33:F_33 = 1)
        # I2_3D = I2_2D + I1_2D  (derived from C_33=1, C_i3=0)
        # dim_phys = 3, I2_ref = 3
        I1_phys = I1 + sp.Integer(1)
        I2_phys = I2 + I1
        dim_phys = 3
        I2_reference = sp.Integer(3)
    else:
        I1_phys = I1
        I2_phys = I2
        dim_phys = dim
        I2_reference = sp.Rational(dim * (dim - 1), 2)  # = 3 for dim=3

    return (
        c1 * (J ** sp.Rational(-2, 3) * I1_phys - dim_phys)
        + c2 * (J ** sp.Rational(-4, 3) * I2_phys - I2_reference)
        + kappa * sp.log(J) ** 2 / 2
    )


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        F = gen.variable(
            gen.Identity(dim) + gen.grad(u),
            name="F",
            qualifier=gen.DEFORMATION_GRADIENT,
        )
        system.add_energy("", _modified_mooney_rivlin_energy(F, dim), fields=(u,), variables=(F,))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "modified_mooney_rivlin",
    systems,
    elements=gen.sfem_supported_element_types(),
    op_name="GeneratedModifiedMooneyRivlin",
    parameter_defaults=(("c1", 1.0), ("c2", 1.0), ("kappa", 1.0)),
    matrix_formats=("bsr",),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
