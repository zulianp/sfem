"""The loperand, and the metric that can be isolated out of it.

SFEM's own name for the fused object is `loperand` -- see
`tet4_linear_elasticity_loperand`, which takes the displacement gradient in
place and returns the flux already contracted with the geometry, so that the
element vector is nothing but differences of its entries.

Written that way an affine element's operator is a linear map from the
reference-gradient coefficients to the outputs, and the question worth asking
about geometry is structural: **is that map's matrix free of the gradient, and
is it symmetric?**  If it is, the whole of it -- Jacobian, quadrature weight,
and whatever material parameters it contains -- can be precomputed per element
and the kernel becomes a contraction against a cached symmetric object.

This replaces asking whether the flux happens to be a uniform scalar multiple
of the field gradient.  That condition is true of the Laplacian and of nothing
else, so it reached exactly one operator by construction and could not have
found another.  Matching the loperand finds the same case without being told
about it, and reports what else it finds:

    laplace, scalar_potential   3x3 symmetric, 6 components, carrying kappa
    linear_elasticity           9x9 symmetric, 45 components, carrying mu, lambda
    stokes                      12x12 symmetric, 78 components, carrying mu
    saint_venant_kirchhoff      not isolatable: the map depends on the gradient
    neohookean_ogden            not isolatable: the map depends on the gradient
    two_phase_flow              isolatable but not symmetric, and carrying the state

So whether to cache is a cost question this makes answerable, not a capability
question to be decided in advance: six components against the nine adjugate
entries and a determinant is a win, forty-five is not.  The matcher reports the
shape and the count; the caller chooses.
"""

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class IsolatedLoperand:
    """The matrix an affine element's loperand contracts the gradient with.

    ``matrix`` is ``(n_field_components * dim)`` square, in row-major order.
    ``carries`` names the non-geometry symbols in it -- material parameters,
    and anything else the flux depends on -- because a matrix carrying the
    solution state is element-varying in a way a cache cannot follow, while one
    carrying only parameters is exactly what a cache is for.
    """

    dim: int
    n_field_components: int
    matrix: tuple
    symmetric: bool
    carries: tuple

    @property
    def order(self):
        return self.n_field_components * self.dim

    @property
    def components(self):
        """How many numbers a cache would have to hold per element."""
        order = self.order
        return order * (order + 1) // 2 if self.symmetric else order * order

    def entry(self, row, column):
        return self.matrix[row * self.order + column]


def adjugate_symbols(dim):
    """The adjugate, as the symbols an isolated matrix is expressed in."""
    return sp.Matrix(dim, dim, lambda i, j: sp.Symbol("adj%d_%d" % (i, j)))


def determinant_symbol():
    return sp.Symbol("det")


def gradient_metric_matrix(dim):
    """`FFF` in those symbols: `adj adj^T / det`, the symmetric gradient metric."""
    adjugate = adjugate_symbols(dim)
    return (adjugate * adjugate.T) / determinant_symbol()


def loperand_matrix(flux_form):
    """The loperand's matrix, or ``None`` when the map is not linear.

    The physical gradient of a P1 simplex is ``G adj / det`` with ``G`` the
    reference-gradient coefficients, and the loperand contracts the flux back
    through the adjugate.  Differentiating with respect to ``G`` gives the map;
    if the result still mentions ``G`` the operator is not linear in the
    gradient and there is nothing to isolate.
    """
    dim = flux_form.dim
    n_field_components = flux_form.n_field_components
    adjugate = adjugate_symbols(dim)
    determinant = determinant_symbol()
    coefficients = sp.Matrix(
        n_field_components,
        dim,
        lambda component, direction: sp.Symbol("g%d_%d" % (component, direction)),
    )
    physical = {}
    for component in range(n_field_components):
        for direction in range(dim):
            physical[flux_form.gradient[component * dim + direction]] = (
                sum(
                    coefficients[component, other] * adjugate[other, direction]
                    for other in range(dim)
                )
                / determinant
            )
    flux = [sp.sympify(entry).subs(physical) for entry in flux_form.flux]
    contracted = [
        sum(
            flux[component * dim + k] * adjugate[direction, k] for k in range(dim)
        )
        for component in range(n_field_components)
        for direction in range(dim)
    ]
    unknowns = [
        coefficients[component, direction]
        for component in range(n_field_components)
        for direction in range(dim)
    ]
    matrix = sp.Matrix(
        len(contracted),
        len(unknowns),
        lambda row, column: sp.diff(contracted[row], unknowns[column]),
    )
    if any(entry.has(*unknowns) for entry in matrix):
        return None
    symmetric = sp.simplify(matrix - matrix.T) == sp.zeros(*matrix.shape)
    geometry = {str(symbol) for symbol in adjugate} | {str(determinant)}
    carries = sorted(
        {
            str(symbol)
            for entry in matrix
            for symbol in entry.free_symbols
            if str(symbol) not in geometry
        }
    )
    return IsolatedLoperand(
        dim=dim,
        n_field_components=n_field_components,
        matrix=tuple(matrix),
        symmetric=symmetric,
        carries=tuple(carries),
    )


def gradient_metric_scale(isolated):
    """The scalar ``s`` when the isolated matrix is ``s * FFF``, else ``None``.

    This is the shape the six-stream metric ABI carries: one symmetric
    ``dim x dim`` object, so it applies to a single-component field whose map is
    a multiple of the gradient metric.  Written as a question about the isolated
    matrix rather than about the flux, so that the answer follows from the
    operator's structure and a material that reaches the same structure another
    way reaches the same kernel.
    """
    if isolated is None or not isolated.symmetric:
        return None
    if isolated.n_field_components != 1:
        return None
    metric = gradient_metric_matrix(isolated.dim)
    scale = None
    for row in range(isolated.dim):
        for column in range(isolated.dim):
            reference = metric[row, column]
            entry = isolated.entry(row, column)
            if reference == 0:
                if sp.simplify(entry) != 0:
                    return None
                continue
            ratio = sp.simplify(sp.together(entry / reference))
            if ratio.free_symbols & metric.free_symbols:
                return None
            if scale is None:
                scale = ratio
            elif sp.simplify(scale - ratio) != 0:
                return None
    return scale
