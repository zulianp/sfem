#!/usr/bin/env python3
import argparse
from pathlib import Path
from itertools import product

import sympy as sp

from sfem import gen


DEFAULT_POLYNOMIAL_ORDER = 1

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
    padded = exponent + (0,) * (3 - len(exponent))
    suffix = (
        "_".join(str(power) for power in padded)
        if any(power > 9 for power in padded)
        else "".join(str(power) for power in padded)
    )
    return "t%d_%s" % (component, suffix)


def _polynomial_traction(dim, order):
    x = gen.SpatialCoordinate()
    exponents = tuple(_monomial_exponents(dim, order))
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


def _parameter_defaults(order):
    defaults = []
    for component in range(3):
        for exponent in _monomial_exponents(3, order):
            defaults.append((_coefficient_name(component, exponent), 0.0))
    return tuple(defaults)


def _validate_polynomial_order(order):
    order = int(order)
    if order < 0:
        raise ValueError("polynomial order must be non-negative")
    return order


def _build_system(dim, polynomial_order=DEFAULT_POLYNOMIAL_ORDER):
    polynomial_order = _validate_polynomial_order(polynomial_order)
    system = gen.EquationSystemBuilder(dim)
    with gen.geometric_dimension_context(dim):
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        v = gen.TestFunction(V, name="u_test")
        traction = _polynomial_traction(dim, polynomial_order)

        system.add_residual("", gen.inner(traction, v) * gen.ds, fields=(u,))
    return system.build()


def create_material(polynomial_order=DEFAULT_POLYNOMIAL_ORDER):
    polynomial_order = _validate_polynomial_order(polynomial_order)
    systems = gen.EquationSystems()
    for dim in (2, 3):
        systems.add(_build_system(dim, polynomial_order))
    return gen.CodeGenerator(
        "neumann_general",
        systems,
        op_name="GeneratedNeumannGeneral",
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
        parameter_defaults=_parameter_defaults(polynomial_order),
    )


material = create_material()


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--polynomial-order", type=int, default=DEFAULT_POLYNOMIAL_ORDER
    )
    args, remaining = parser.parse_known_args(argv)
    gen.run(
        create_material(args.polynomial_order),
        Path(__file__).with_name("generated") / material.name,
        argv=remaining,
    )


if __name__ == "__main__":
    main()
