"""The element kernel for an affine, lowest-order simplex, in closed form.

On TRI3 and TET4 the basis gradients are constant over the element, so a
quadrature loop has one trip and the per-point data it would index never
varies.  The rule for this family is to evaluate in closed form instead: no
quadrature loop, no reference-basis tables, and the arithmetic expanded through
a compact physical gradient.

"Compact physical gradient" is the whole of it.  For a P1 field the physical
gradient is ``grad_d = u_{d+1} - u_0``, and the element contribution is that
gradient carried through the metric and scattered back:

    q  = FFF * grad          the flux
    e_0     = -sum(q)        because the reference gradients sum to zero
    e_{d+1} = q_d

That intermediate is what keeps it small.  Deriving the same contraction
directly in the degrees of freedom, without forming the gradient, gives 47
operations in three dimensions against 21 this way -- and expanding the algebra
before CSE gives 50 with fifteen temporaries, worse than either.  The emitted
2D kernel has always used the gradient form; the 3D one was a transcription of
a hand-written kernel that did not, which is where the difference was hiding.

The expressions are symbolic and named abstractly.  The emitter binds ``fff``
and ``u`` to whatever the ABI calls them, which is the plan layer's business to
decide and emission's to spell.
"""

from dataclasses import dataclass

import sympy as sp

from codegen.framework.plans.form_transformations import (
    symmetric_metric_component_count,
    symmetric_metric_storage_component_index,
)

#: How the metric components are packed in the ABI.  Not a default worth
#: guessing: `symmetric_metric_component_index` packs upper-column-major while
#: the geometry streams are written upper-row-major, and the two agree only on
#: the diagonal.  Taking the wrong one silently permutes the metric, which
#: still compiles, still runs, and gives wrong answers on any element whose
#: metric is not isotropic.
DEFAULT_COMPONENT_ORDER = "upper_row_major"


@dataclass(frozen=True)
class AffineElementKernelPlan:
    """The closed-form element contribution, as temporaries and outputs."""

    dim: int
    #: ``(symbol, expression)`` in evaluation order, already reduced.
    temporaries: tuple
    #: One expression per shape function, in node order.
    outputs: tuple

    @property
    def n_shape(self):
        return self.dim + 1


def metric_symbols(dim):
    """The symmetric metric components, in the storage order the ABI uses."""
    return sp.symbols("fff0:%d" % symmetric_metric_component_count(dim))


def dof_symbols(dim):
    """One symbol per node of the simplex."""
    return sp.symbols("u0:%d" % (dim + 1))


def metric_matrix(dim, components=None, order=DEFAULT_COMPONENT_ORDER):
    """The metric as a symmetric matrix, from its packed components."""
    components = metric_symbols(dim) if components is None else components
    return sp.Matrix(
        dim,
        dim,
        lambda row, column: components[
            symmetric_metric_storage_component_index(dim, row, column, order)
        ],
    )


def p1_simplex_metric_apply_plan(
    dim, temporary_prefix="t", order=DEFAULT_COMPONENT_ORDER
):
    """The element apply for a P1 simplex whose geometry is a cached metric.

    Returns the plan; the caller spells it.  CSE runs on the *factored*
    contraction -- see the module docstring for why expanding first is a
    pessimisation rather than a simplification.
    """
    dim = int(dim)
    metric = metric_matrix(dim, order=order)
    dofs = dof_symbols(dim)
    gradient = sp.Matrix([dofs[d + 1] - dofs[0] for d in range(dim)])
    flux = metric * gradient
    outputs = [-sum(flux)] + [flux[d] for d in range(dim)]
    temporaries, reduced = sp.cse(
        outputs, symbols=sp.numbered_symbols(temporary_prefix)
    )
    return AffineElementKernelPlan(
        dim=dim,
        temporaries=tuple(temporaries),
        outputs=tuple(reduced),
    )
