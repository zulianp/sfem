"""What a boundary kernel's coefficients actually use.

A Neumann-style boundary form may or may not depend on the physical
coordinates, and may or may not depend on the current field values.  Each answer
decides real structure in the emitted kernel: whether coordinates are gathered
and passed, whether the current solution is interpolated to the quadrature
point, and which arguments the local and mesh signatures carry.  Per component
it also decides whether that component's expression needs either at all.

The analysis is a scan of free symbols, and deciding what a kernel reads is
planning, not emission.  It was being done inside
``emitters/boundary_codegen.py`` in four places, twice as whole-set queries and
twice per component, so the same question was answered with the same code in
both a planning shape and an emission shape.
"""

from dataclasses import dataclass

import sympy as sp


def coordinate_symbols(physical_dim):
    """The ``x0..xn`` symbols a boundary form may reference."""
    return tuple(sp.Symbol("x%d" % d) for d in range(int(physical_dim)))


@dataclass(frozen=True)
class BoundaryCoefficientUsage:
    """Which coordinate and current-value symbols the coefficients reach for."""

    coordinates: tuple
    current: tuple
    component_uses_coordinates: tuple
    component_uses_current: tuple

    @property
    def uses_coordinates(self):
        return bool(self.coordinates)

    @property
    def uses_current(self):
        return bool(self.current)

    def to_dict(self):
        return {
            "coordinates": [str(s) for s in self.coordinates],
            "current": [str(s) for s in self.current],
            "component_uses_coordinates": list(self.component_uses_coordinates),
            "component_uses_current": list(self.component_uses_current),
        }


def boundary_coefficient_usage(coefficients, coordinate_candidates, current_candidates):
    """Analyse ``coefficients`` once, for the whole form and per component.

    Both candidate tuples are ordered, and both results follow that order rather
    than the order symbols happen to be discovered in, so the emitted argument
    lists are stable.  Pass ``coordinate_symbols(dim)`` for the first when
    starting from a dimension.
    """
    coefficients = tuple(coefficients)
    coordinate_candidates = tuple(coordinate_candidates)
    coordinate_set = set(coordinate_candidates)
    current_candidates = tuple(current_candidates)
    current_set = set(current_candidates)

    per_component_coordinates = []
    per_component_current = []
    used_coordinates = set()
    used_current = set()
    for coefficient in coefficients:
        free = sp.sympify(coefficient).free_symbols
        hit_coordinates = free.intersection(coordinate_set)
        hit_current = free.intersection(current_set)
        used_coordinates.update(hit_coordinates)
        used_current.update(hit_current)
        per_component_coordinates.append(bool(hit_coordinates))
        per_component_current.append(bool(hit_current))

    return BoundaryCoefficientUsage(
        coordinates=tuple(s for s in coordinate_candidates if s in used_coordinates),
        current=tuple(s for s in current_candidates if s in used_current),
        component_uses_coordinates=tuple(per_component_coordinates),
        component_uses_current=tuple(per_component_current),
    )
