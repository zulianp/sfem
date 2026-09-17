"""Lame solution for an internally pressurized, externally free thick sphere."""

import numpy as np


def constants(inner_radius, outer_radius, pressure):
    a3 = inner_radius ** 3
    b3 = outer_radius ** 3
    uniform_stress = pressure * a3 / (b3 - a3)
    inverse_cubic_stress = pressure * a3 * b3 / (b3 - a3)
    return uniform_stress, inverse_cubic_stress


def radial_displacement(radius, inner_radius, outer_radius, pressure, mu, lmbda):
    uniform, inverse_cubic = constants(inner_radius, outer_radius, pressure)
    bulk = lmbda + 2.0 * mu / 3.0
    return uniform * radius / (3.0 * bulk) + inverse_cubic / (4.0 * mu * radius ** 2)


def polar_stress(radius, inner_radius, outer_radius, pressure):
    uniform, inverse_cubic = constants(inner_radius, outer_radius, pressure)
    radial = uniform - inverse_cubic / radius ** 3
    hoop = uniform + inverse_cubic / (2.0 * radius ** 3)
    return np.stack((radial, hoop), axis=-1)


def displacement(points, inner_radius, outer_radius, pressure, mu, lmbda):
    points = np.asarray(points, dtype=np.float64)
    radius = np.linalg.norm(points, axis=-1)
    radial = radial_displacement(radius, inner_radius, outer_radius, pressure, mu, lmbda)
    return radial[..., None] * points / radius[..., None]


def octant_pressure_resultant(inner_radius, pressure):
    component = pressure * np.pi * inner_radius ** 2 / 4.0
    return np.full(3, component)
