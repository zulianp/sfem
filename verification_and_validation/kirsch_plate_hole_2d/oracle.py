"""Infinite-plane Kirsch solution restricted to a finite quarter annulus."""

import numpy as np


def poisson_ratio(mu, lmbda):
    return lmbda / (2.0 * (mu + lmbda))


def displacement(points, hole_radius, tension, mu, lmbda):
    points = np.asarray(points, dtype=np.float64)
    radius = np.linalg.norm(points, axis=-1)
    theta = np.arctan2(points[..., 1], points[..., 0])
    nu = poisson_ratio(mu, lmbda)
    a2 = hole_radius ** 2
    a4 = a2 ** 2
    factor = tension / (4.0 * mu)
    radial = factor * (
        (1.0 - 2.0 * nu) * radius + a2 / radius
        + np.cos(2.0 * theta) * (radius + 4.0 * (1.0 - nu) * a2 / radius - a4 / radius ** 3)
    )
    angular = factor * np.sin(2.0 * theta) * (
        -radius + (4.0 * nu - 2.0) * a2 / radius - a4 / radius ** 3
    )
    c = np.cos(theta)
    s = np.sin(theta)
    return np.stack((radial * c - angular * s, radial * s + angular * c), axis=-1)


def polar_stress(points, hole_radius, tension):
    points = np.asarray(points, dtype=np.float64)
    radius = np.linalg.norm(points, axis=-1)
    theta = np.arctan2(points[..., 1], points[..., 0])
    ratio2 = (hole_radius / radius) ** 2
    ratio4 = ratio2 ** 2
    c = np.cos(2.0 * theta)
    s = np.sin(2.0 * theta)
    radial = 0.5 * tension * ((1.0 - ratio2) + (1.0 - 4.0 * ratio2 + 3.0 * ratio4) * c)
    hoop = 0.5 * tension * ((1.0 + ratio2) - (1.0 + 3.0 * ratio4) * c)
    shear = -0.5 * tension * (1.0 + 2.0 * ratio2 - 3.0 * ratio4) * s
    return np.stack((radial, hoop, shear), axis=-1)
