#!/usr/bin/env python3
from pathlib import Path

import sympy as sp

from sfem import gen


mu, lmbda = sp.symbols("mu lmbda")


def strain_energy(F):
    dim = F.rows
    log_j = sp.log(F.det())
    return (
        mu * (gen.matrix_inner(F, F) - dim) / 2
        - mu * log_j
        + lmbda * log_j**2 / 2
    )


material = gen.HyperelasticMaterial(
    "neohookean_ogden",
    strain_energy,
)


if __name__ == "__main__":
    gen.run(material, Path(__file__).with_name("generated") / material.name)
