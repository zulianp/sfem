"""The inexact Hessian action: quadrature removed by an L2 projection.

The exact matrix-free Hessian action of a hyperelastic operator is

    (H h)_{i,p} = sum_q w_q * S_{ikmn}(F(xi_q)) * dh_k/dxi_n * dphi_p/dxi_m

with `S` the material tangent already pulled back through the geometry and
weighted -- the loperand of `plans.loperand`, for a form whose flux is
nonlinear.  `S` varies over the element because `F` does, which is what forces
the quadrature loop.

Project `S` onto the constants -- its L2 projection onto P0, which is its
average over the element -- and the loop separates:

    (H h)_{i,p} ~= Sbar_{ikmn} * sum_j h_{k,j} * Wbar_{jmpn}

    Wbar_{jmpn} = integral over the reference cell of
                  dphi_j/dxi_m * dphi_p/dxi_n

`Sbar` is one small object per element, computed once from the state.  `Wbar`
is a pure property of the element, integrated exactly here, at generation time,
and never computed at run time at all.  The quadrature loop is gone from the
kernel.

**Where the inexactness is, and where it is not.**  On an affine simplex the
deformation gradient is constant over the element, so `S` is constant, so its
projection onto the constants is `S` itself and the kernel is *exact*.  The
approximation appears only where the state genuinely varies across the cell --
a curved or higher-order element -- and there it is a one-term projection whose
error has to be measured rather than assumed.  That split is what makes this
testable: the affine simplex is the case where the machinery must reproduce the
exact kernel, and it is the gate on everything the other elements then do.

**The compression, and the trap in it.**  `Wbar` is small in a specific way:
whatever the element, it takes at most ten distinct rational values, and most
of them are zero.

    TET4    144 entries    3 distinct   108 zero
    TRI3     36 entries    3 distinct    20 zero
    HEX8    576 entries   10 distinct     0 zero
    TET10   900 entries    9 distinct   519 zero

The trap is to store those values in a runtime table and index it, which is
what compresses the *memory* and destroys the *arithmetic*: the constants stop
being constants, every multiply by zero survives into the emitted code, and
nothing downstream can fold them.  Measured on the same contraction:

    TET4    constants folded    31 temporaries,   334 operations
            runtime table      229 temporaries,  1406 operations
    HEX8    constants folded  1500 temporaries,  4544 operations
            runtime table      399 temporaries,  6231 operations

So `Wbar` is folded, not tabulated.  Its compression is real and is spent at
generation time, where a rational that appears ninety-six times costs nothing
to repeat and a zero costs nothing at all.
"""

from dataclasses import dataclass
from functools import lru_cache
import itertools

import sympy as sp

from codegen.framework.fem.reference_basis import reference_basis


@dataclass(frozen=True)
class ReferenceGradientProduct:
    """`Wbar`, and what its entries turn out to be.

    Indexed ``[node, direction, node, direction]`` and flattened row-major, the
    order every four-tensor in the framework uses.
    """

    element_type: str
    dim: int
    n_nodes: int
    entries: tuple

    def entry(self, node, direction, other_node, other_direction):
        stride = self.n_nodes * self.dim
        return self.entries[
            ((node * self.dim + direction) * stride)
            + other_node * self.dim
            + other_direction
        ]

    @property
    def n_entries(self):
        return len(self.entries)

    @property
    def distinct_values(self):
        """The distinct rationals it takes, in a stable order."""
        return tuple(sorted({sp.nsimplify(e) for e in self.entries}, key=sp.default_sort_key))

    @property
    def n_zero(self):
        return sum(1 for entry in self.entries if entry == 0)

    @property
    def is_major_symmetric(self):
        """`Wbar_{jmpn} == Wbar_{pnjm}`, which it is for every element here.

        It is an integral of a product of two gradients, so swapping the two
        (node, direction) pairs swaps the factors and nothing else.
        """
        return all(
            self.entry(j, m, p, n) == self.entry(p, n, j, m)
            for j in range(self.n_nodes)
            for m in range(self.dim)
            for p in range(self.n_nodes)
            for n in range(self.dim)
        )


@lru_cache(maxsize=None)
def reference_gradient_product(element_type):
    """`Wbar` for this element, integrated exactly, or ``None``.

    Cached because it is a property of the element and the integration is the
    expensive part of the whole plan -- once per element per process, never per
    material and never per call.
    """
    basis = reference_basis(element_type)
    if basis is None:
        return None
    gradients = basis.gradients()
    entries = []
    for node, direction, other_node, other_direction in itertools.product(
        range(basis.n_nodes), range(basis.dim), range(basis.n_nodes), range(basis.dim)
    ):
        entries.append(
            basis.integrate(
                gradients[node][direction] * gradients[other_node][other_direction]
            )
        )
    return ReferenceGradientProduct(
        element_type=basis.element_type,
        dim=basis.dim,
        n_nodes=basis.n_nodes,
        entries=tuple(entries),
    )


#: The elements on which the projection onto the constants is not an
#: approximation at all, because the deformation gradient does not vary over
#: the cell.  These are the elements the inexact kernel must agree with the
#: exact one on, which is what makes the rest of it checkable.
EXACT_ON_ELEMENTS = ("TRI3", "TET4")


def projection_is_exact(element_type):
    """Whether the P0 projection loses nothing on this element."""
    return str(element_type).upper() in EXACT_ON_ELEMENTS


@dataclass(frozen=True)
class InexactHessianPlan:
    """What an inexact Hessian-action kernel emits.

    `tangent_components` is how many numbers the projected tangent takes per
    element, using its major symmetry: a `dim^2` square, so 45 in three
    dimensions and 10 in two.
    """

    element_type: str
    dim: int
    n_nodes: int
    reference: ReferenceGradientProduct
    exact: bool

    @property
    def tangent_components(self):
        order = self.dim * self.dim
        return order * (order + 1) // 2

    def tangent_index(self, i, k, m, n):
        """Where `S_{ikmn}` sits in the packed tangent, by major symmetry."""
        row, column = i * self.dim + k, m * self.dim + n
        if row > column:
            row, column = column, row
        order = self.dim * self.dim
        return row * order - row * (row - 1) // 2 + (column - row)

    def action(self, tangent, increment):
        """`(H h)_{i,p}`, as expressions in the packed tangent and increment.

        `tangent` is indexed by the packed index above and `increment` by
        `[component][node]`.  The reference tensor's entries go in as the
        rationals they are, so a zero removes its term here rather than
        surviving as a multiply in the emitted code.
        """
        outputs = []
        for component in range(self.dim):
            for node in range(self.n_nodes):
                total = sp.Integer(0)
                for other, m, n in itertools.product(
                    range(self.dim), range(self.dim), range(self.dim)
                ):
                    weight = sum(
                        (
                            increment[other][j]
                            * self.reference.entry(j, m, node, n)
                            for j in range(self.n_nodes)
                            if self.reference.entry(j, m, node, n) != 0
                        ),
                        sp.Integer(0),
                    )
                    if weight == 0:
                        continue
                    total += (
                        tangent[self.tangent_index(component, other, m, n)] * weight
                    )
                outputs.append(sp.expand(total))
        return tuple(outputs)


def inexact_hessian_plan(element_type):
    """That plan, or ``None`` for an element with no symbolic basis."""
    reference = reference_gradient_product(element_type)
    if reference is None:
        return None
    return InexactHessianPlan(
        element_type=reference.element_type,
        dim=reference.dim,
        n_nodes=reference.n_nodes,
        reference=reference,
        exact=projection_is_exact(reference.element_type),
    )
