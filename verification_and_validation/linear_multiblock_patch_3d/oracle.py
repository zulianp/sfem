"""Closed-form two-material extension with continuous interface traction."""

import numpy as np


def response(material):
    left_modulus = float(material["left_lambda"]) + 2.0 * float(material["left_mu"])
    right_modulus = float(material["right_lambda"]) + 2.0 * float(material["right_mu"])
    stress = 1.0
    left_strain = stress / left_modulus
    right_strain = stress / right_modulus
    total_displacement = 0.5 * (left_strain + right_strain)
    energy = 0.25 * stress * (left_strain + right_strain)
    return {
        "stress": stress,
        "left_strain": left_strain,
        "right_strain": right_strain,
        "total_displacement": total_displacement,
        "energy": energy,
    }


def displacement(points, material):
    values = response(material)
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    ux = np.where(
        x <= 0.5,
        values["left_strain"] * x,
        0.5 * values["left_strain"] + values["right_strain"] * (x - 0.5),
    )
    result = np.zeros_like(points)
    result[:, 0] = ux
    return result
