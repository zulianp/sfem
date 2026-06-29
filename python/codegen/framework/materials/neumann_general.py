#!/usr/bin/env python3
from pathlib import Path
from itertools import product

import sympy as sp

from sfem import gen


element = gen.VectorElement("Lagrange", degree=1)
V = gen.FunctionSpace(element)


def _monomial_exponents(dim, order):
    candidates = product(range(order + 1), repeat=dim)
    for total in range(order + 1):
        for exponent in candidates:
            if sum(exponent) == total:
                yield exponent
        candidates = product(range(order + 1), repeat=dim)


def _coefficient_name(component, exponent):
    if sum(exponent) == 0:
        return "t%d" % component
    suffix = "".join(str(power) for power in exponent + (0,) * (3 - len(exponent)))
    return "t%d_%s" % (component, suffix)


def _polynomial_traction(dim):
    x = gen.SpatialCoordinate()
    exponents = tuple(_monomial_exponents(dim, 3))
    traction = []
    for component in range(dim):
        value = sp.S.Zero
        for exponent in exponents:
            monomial = sp.S.One
            for coordinate, power in zip(x, exponent):
                if power:
                    monomial *= coordinate**power
            value += (
                gen.material_parameter(_coefficient_name(component, exponent))
                * monomial
            )
        traction.append(value)
    return sp.Matrix(traction)


def _parameter_defaults():
    defaults = []
    for component in range(3):
        for exponent in _monomial_exponents(3, 3):
            defaults.append((_coefficient_name(component, exponent), 0.0))
    return tuple(defaults)


def _build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        v = gen.TestFunction(V, name="u_test")
        traction = _polynomial_traction(dim)

        system.add_residual("", gen.inner(traction, v) * gen.ds, fields=(u,))
    return system.build()


systems = gen.EquationSystems()
for dim in (2, 3):
    systems.add(_build_system(dim))


material = gen.CodeGenerator(
    "neumann",
    systems,
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
    parameter_defaults=_parameter_defaults(),
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
