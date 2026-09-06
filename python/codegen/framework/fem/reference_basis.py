"""Symbolic reference bases, and exact integration over the reference cell.

The framework's reference data is numeric: basis values and gradients sampled
at quadrature points, which is what a quadrature loop needs.  Removing the
quadrature loop needs the other thing -- the basis as an expression, so that
integrals of products of its gradients can be taken exactly, once, at
generation time.

That is all this provides.  It is deliberately small: the bases below are the
ones the framework generates for, written out rather than derived, and the
integration is `sympy.integrate` over the reference cell's own limits.  Nothing
here runs per element or per call.
"""

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp


#: The reference coordinates, in the order the element's limits use them.
REFERENCE_COORDINATES = sp.symbols("xi0 xi1 xi2")


@dataclass(frozen=True)
class ReferenceBasis:
    """One element's basis over its reference cell."""

    element_type: str
    dim: int
    #: One expression per node, in the element's own local node order.
    functions: tuple
    #: `sympy.integrate` limits, innermost first, describing the cell.
    limits: tuple

    @property
    def n_nodes(self):
        return len(self.functions)

    @property
    def coordinates(self):
        return REFERENCE_COORDINATES[: self.dim]

    def gradients(self):
        """`dphi_i/dxi_m`, indexed [node][direction]."""
        return tuple(
            tuple(sp.diff(function, coordinate) for coordinate in self.coordinates)
            for function in self.functions
        )

    def integrate(self, expression):
        """That expression over the reference cell, exactly."""
        for limit in self.limits:
            expression = sp.integrate(expression, limit)
        return sp.nsimplify(sp.simplify(expression))

    @property
    def measure(self):
        return self.integrate(sp.Integer(1))


def _simplex(dim):
    x, y, z = REFERENCE_COORDINATES
    if dim == 2:
        return [1 - x - y, x, y], ((y, 0, 1 - x), (x, 0, 1))
    return [1 - x - y - z, x, y, z], ((z, 0, 1 - x - y), (y, 0, 1 - x), (x, 0, 1))


def _tensor_product(dim):
    x, y, z = REFERENCE_COORDINATES
    coordinates = (x, y, z)[:dim]
    functions = []
    # Lexicographic in the last coordinate first, which is the corner order the
    # framework's tensor-product elements use.
    for corner in _corners(dim):
        term = sp.Integer(1)
        for axis, bit in enumerate(corner):
            term *= coordinates[axis] if bit else (1 - coordinates[axis])
        functions.append(sp.expand(term))
    limits = tuple((coordinate, 0, 1) for coordinate in reversed(coordinates))
    return functions, limits


def _corners(dim):
    if dim == 2:
        return ((0, 0), (1, 0), (1, 1), (0, 1))
    return (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    )


def _tet10():
    x, y, z = REFERENCE_COORDINATES
    barycentric = [1 - x - y - z, x, y, z]
    functions = [
        coordinate * (2 * coordinate - 1) for coordinate in barycentric
    ]
    # SFEM's tet10 edge order: (0,1) (1,2) (0,2) (0,3) (1,3) (2,3).
    for first, second in ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)):
        functions.append(4 * barycentric[first] * barycentric[second])
    limits = ((z, 0, 1 - x - y), (y, 0, 1 - x), (x, 0, 1))
    return [sp.expand(function) for function in functions], limits


_BASIS_BY_ELEMENT = {
    "TRI3": lambda: (2,) + _simplex(2),
    "TET4": lambda: (3,) + _simplex(3),
    "QUAD4": lambda: (2,) + _tensor_product(2),
    "HEX8": lambda: (3,) + _tensor_product(3),
    "TET10": lambda: (3,) + _tet10(),
}


def supported_elements():
    """The elements a reference basis is written out for."""
    return tuple(sorted(_BASIS_BY_ELEMENT))


@lru_cache(maxsize=None)
def reference_basis(element_type):
    """That element's basis, or ``None`` where none is written out.

    ``None`` rather than an exception: an element without a symbolic basis
    simply cannot take the paths that need one, and the caller decides what to
    do about it.
    """
    build = _BASIS_BY_ELEMENT.get(str(element_type).upper())
    if build is None:
        return None
    dim, functions, limits = build()
    return ReferenceBasis(
        element_type=str(element_type).upper(),
        dim=dim,
        functions=tuple(functions),
        limits=tuple(limits),
    )
