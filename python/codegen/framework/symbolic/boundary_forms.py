from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class BoundaryIntegral:
    integrand: object
    measure: str = "ds"

    def __post_init__(self):
        measure = str(self.measure)
        if measure != "ds":
            raise ValueError("only boundary measure 'ds' is currently supported")
        object.__setattr__(self, "integrand", sp.sympify(self.integrand))
        object.__setattr__(self, "measure", measure)


@dataclass(frozen=True)
class Measure:
    name: str

    def __post_init__(self):
        name = str(self.name)
        if name not in ("dx", "ds"):
            raise ValueError("unsupported measure '%s'" % name)
        object.__setattr__(self, "name", name)

    def __call__(self, integrand):
        if self.name == "ds":
            return BoundaryIntegral(integrand, self.name)
        return sp.sympify(integrand)

    def __rmul__(self, integrand):
        return self(integrand)


dx = Measure("dx")
ds = Measure("ds")


def is_boundary_integral(value):
    return isinstance(value, BoundaryIntegral)


def integral_measure(value):
    if isinstance(value, BoundaryIntegral):
        return value.measure
    return "dx"


def integral_integrand(value):
    if isinstance(value, BoundaryIntegral):
        return value.integrand
    return value


__all__ = [
    "BoundaryIntegral",
    "Measure",
    "dx",
    "ds",
    "integral_integrand",
    "integral_measure",
    "is_boundary_integral",
]
