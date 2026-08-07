#!/usr/bin/env python3
"""Manufactured steady Stokes cases used by the verification scripts.

The paper-specific cases come from HAL cea-02434556, section 3:
"FVCA8 benchmark for the Stokes and Navier-Stokes equations with the
TrioCFD code - benchmark session".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class StokesCase:
    name: str
    dim: int
    velocity: Callable[..., Tuple[Array, ...]]
    pressure: Callable[..., Array]
    forcing: Callable[..., Tuple[Array, ...]]
    description: str = ""
    paper_section: str = ""
    viscosity: float = 1.0
    expected_velocity_order: str = ""
    expected_pressure_order: str = ""


def _bercovier_u1(x: Array, y: Array) -> Array:
    return 256.0 * x * x * (x - 1.0) * (x - 1.0) * y * (y - 1.0) * (2.0 * y - 1.0)


def _bercovier_laplacian_core(x: Array, y: Array) -> Array:
    return (
        x * x * (x - 1.0) * (x - 1.0) * (12.0 * y - 6.0)
        + y * (y - 1.0) * (2.0 * y - 1.0) * (12.0 * x * x - 12.0 * x + 2.0)
    )


def _bercovier_engelman_velocity(x: Array, y: Array) -> Tuple[Array, Array]:
    return _bercovier_u1(x, y), -_bercovier_u1(y, x)


def _bercovier_engelman_pressure(x: Array, y: Array) -> Array:
    return (x - 0.5) * (y - 0.5)


def _bercovier_engelman_forcing(mu: float, x: Array, y: Array) -> Tuple[Array, Array]:
    fx = -mu * 256.0 * _bercovier_laplacian_core(x, y) + (y - 0.5)
    fy = mu * 256.0 * _bercovier_laplacian_core(y, x) + (x - 0.5)
    return fx, fy


def _taylor_green_velocity(x: Array, y: Array, z: Array) -> Tuple[Array, Array, Array]:
    two_pi = 2.0 * np.pi
    sx = np.sin(two_pi * x)
    sy = np.sin(two_pi * y)
    sz = np.sin(two_pi * z)
    cx = np.cos(two_pi * x)
    cy = np.cos(two_pi * y)
    cz = np.cos(two_pi * z)
    return 2.0 * cx * sy * sz, -sx * cy * sz, -sx * sy * cz


def _taylor_green_pressure(x: Array, y: Array, z: Array) -> Array:
    two_pi = 2.0 * np.pi
    return 6.0 * np.pi * np.sin(two_pi * x) * np.sin(two_pi * y) * np.sin(two_pi * z)


def _taylor_green_forcing(mu: float, x: Array, y: Array, z: Array) -> Tuple[Array, Array, Array]:
    two_pi = 2.0 * np.pi
    pi2 = np.pi * np.pi
    sx = np.sin(two_pi * x)
    sy = np.sin(two_pi * y)
    sz = np.sin(two_pi * z)
    cx = np.cos(two_pi * x)
    cy = np.cos(two_pi * y)
    cz = np.cos(two_pi * z)
    fx = (24.0 * mu + 12.0) * pi2 * cx * sy * sz
    fy = 12.0 * (1.0 - mu) * pi2 * sx * cy * sz
    fz = 12.0 * (1.0 - mu) * pi2 * sx * sy * cz
    return fx, fy, fz


def _poly1_velocity(x: Array, y: Array) -> Tuple[Array, Array]:
    ux = x * x * (1 - x) * (1 - x) * 2 * y * (1 - y) * (2 * y - 1)
    uy = y * y * (1 - y) * (1 - y) * 2 * x * (1 - x) * (1 - 2 * x)
    return ux, uy


def _poly1_pressure(x: Array, y: Array) -> Array:
    return x * (1 - x) * (1 - y) - 1.0 / 12.0


def _poly1_forcing(mu: float, x: Array, y: Array) -> Tuple[Array, Array]:
    fx = -mu * (
        4 * y * (1 - y) * (2 * y - 1) * ((1 - 2 * x) * (1 - 2 * x) - 2 * x * (1 - x))
        + 12 * x * x * (1 - x) * (1 - x) * (1 - 2 * y)
    ) + (1 - 2 * x) * (1 - y)
    fy = -mu * (
        4 * x * (1 - x) * (1 - 2 * x) * ((1 - 2 * y) * (1 - 2 * y) - 2 * y * (1 - y))
        + 12 * y * y * (1 - y) * (1 - y) * (2 * x - 1)
    ) - x * (1 - x)
    return fx, fy


def _poly2_velocity(x: Array, y: Array) -> Tuple[Array, Array]:
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    y2 = y * y
    y3 = y2 * y
    y4 = y3 * y
    ux = (x2 - 2 * x3 + x4) * (2 * y - 6 * y2 + 4 * y3)
    uy = -(2 * x - 6 * x2 + 4 * x3) * (y2 - 2 * y3 + y4)
    return ux, uy


def _poly2_pressure(x: Array, y: Array) -> Array:
    return (x + y - 1) / 24.0


def _poly2_forcing(mu: float, x: Array, y: Array) -> Tuple[Array, Array]:
    fx = -mu * (
        (2 - 12 * x + 12 * x * x) * (2 * y - 6 * y * y + 4 * y * y * y)
        + (x * x - 2 * x * x * x + x * x * x * x) * (-12 + 24 * y)
    ) + 1.0 / 24.0
    fy = mu * (
        (2 - 12 * y + 12 * y * y) * (2 * x - 6 * x * x + 4 * x * x * x)
        + (y * y - 2 * y * y * y + y * y * y * y) * (-12 + 24 * x)
    ) + 1.0 / 24.0
    return fx, fy


def _trig_velocity(x: Array, y: Array) -> Tuple[Array, Array]:
    two_pi = 2.0 * np.pi
    ux = np.sin(two_pi * y) * (1 - np.cos(two_pi * x))
    uy = np.sin(two_pi * x) * (np.cos(two_pi * y) - 1)
    return ux, uy


def _trig_pressure(x: Array, y: Array) -> Array:
    two_pi = 2.0 * np.pi
    return two_pi * (np.cos(two_pi * y) - np.cos(two_pi * x))


def _trig_forcing(mu: float, x: Array, y: Array) -> Tuple[Array, Array]:
    two_pi = 2.0 * np.pi
    four_pi2 = 4.0 * np.pi * np.pi
    fx = -four_pi2 * mu * np.sin(two_pi * y) * (2 * np.cos(two_pi * x) - 1) + four_pi2 * np.sin(two_pi * x)
    fy = four_pi2 * mu * np.sin(two_pi * x) * (2 * np.cos(two_pi * y) - 1) - four_pi2 * np.sin(two_pi * y)
    return fx, fy


CASES: Dict[str, StokesCase] = {
    "bercovier_engelman_2d": StokesCase(
        "bercovier_engelman_2d",
        2,
        _bercovier_engelman_velocity,
        _bercovier_engelman_pressure,
        _bercovier_engelman_forcing,
        description="2D Bercovier-Engelman steady Stokes benchmark on [0,1]^2.",
        paper_section="3.1",
        expected_velocity_order="2 on triangular and rectangular meshes",
        expected_pressure_order="1 on triangular meshes, 2 on rectangular meshes",
    ),
    "taylor_green_3d": StokesCase(
        "taylor_green_3d",
        3,
        _taylor_green_velocity,
        _taylor_green_pressure,
        _taylor_green_forcing,
        description="3D Taylor-Green vortex steady Stokes benchmark on [0,1]^3.",
        paper_section="3.2",
        expected_velocity_order="about 1.7-2 on tetrahedral meshes, 2 on hexahedral meshes",
        expected_pressure_order="1 on tetrahedral meshes, 2 on hexahedral meshes",
    ),
    "polynomial_1": StokesCase("polynomial_1", 2, _poly1_velocity, _poly1_pressure, _poly1_forcing),
    "polynomial_2": StokesCase("polynomial_2", 2, _poly2_velocity, _poly2_pressure, _poly2_forcing),
    "trigonometric": StokesCase("trigonometric", 2, _trig_velocity, _trig_pressure, _trig_forcing),
}


def case_by_name(name: str) -> StokesCase:
    try:
        return CASES[name]
    except KeyError as exc:
        raise ValueError("unknown Stokes verification case '%s'" % name) from exc


def describe_cases() -> str:
    lines = []
    for name in sorted(CASES):
        case = CASES[name]
        description = case.description or "legacy local manufactured solution"
        lines.append("%s: %sD, %s" % (name, case.dim, description))
        if case.paper_section:
            lines.append("  paper section: %s, viscosity: %g" % (case.paper_section, case.viscosity))
            lines.append("  expected velocity order: %s" % case.expected_velocity_order)
            lines.append("  expected pressure order: %s" % case.expected_pressure_order)
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_cases())
