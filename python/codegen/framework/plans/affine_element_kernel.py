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


@dataclass(frozen=True)
class ExpandedSimplexMetricPlan:
    """The closed-form mesh loop a lowest-order simplex calls for.

    Carries the element algebra together with the two facts the loop around it
    needs: which field role it reads, and the scale the metric is multiplied
    by.  Both were being decided in emission, which is emission choosing what
    to emit rather than how to spell it.
    """

    kernel: AffineElementKernelPlan
    #: "u" when the form reads the current state, "h" when it reads a
    #: direction.  A form whose flux factors through the metric is linear, so
    #: it is one or the other and never both.
    input_prefix: str
    scale: object

    @property
    def dim(self):
        return self.kernel.dim

    @property
    def n_shape(self):
        return self.kernel.n_shape


def expanded_simplex_metric_plan(
    metric,
    dim,
    n_nodes,
    n_qp,
    n_field_components,
    writes_per_shape,
    reads_current,
    reads_direction,
):
    """That plan, or ``None`` when the shape does not call for it.

    The conditions are the ones the element and the lowered form already
    settle: a cached metric, so the geometry is one symmetric tensor per
    element; a single quadrature point on a simplex of ``dim + 1`` nodes, so
    the basis gradients are constant; one field component, because the compact
    contraction below is written for a scalar; a form that writes per shape,
    so there is an element vector to scatter; and exactly one live field role.

    Asked here rather than in the emitter so that emission has one question to
    ask and no answer to derive.
    """
    if metric is None or n_qp != 1:
        return None
    if dim not in (2, 3) or n_nodes != dim + 1:
        return None
    if int(n_field_components) != 1 or not writes_per_shape:
        return None
    if bool(reads_current) == bool(reads_direction):
        return None
    return ExpandedSimplexMetricPlan(
        kernel=p1_simplex_metric_apply_plan(dim),
        input_prefix="u" if reads_current else "h",
        scale=metric.scale,
    )


def p1_simplex_metric_value_plan(
    dim, temporary_prefix="t", order=DEFAULT_COMPONENT_ORDER
):
    """The element energy for a P1 simplex whose geometry is a cached metric.

    The 0-form of the same operator ``p1_simplex_metric_apply_plan`` gives the
    1-form of.  With ``g`` the reference gradient ``(u_1 - u_0, ...)``, the
    element energy is ``g^T FFF g / 2``: the metric already carries
    ``J^-1 J^-T det J``, so the quadrature weight and the Jacobian are in it and
    there is nothing else to multiply by.

    The two plans agree exactly -- ``E == (1/2) * sum_a e_a u_a`` with ``e`` the
    apply plan's outputs -- which is asserted in the tests rather than assumed
    here, and is what says the 0-form and the 1-form describe one operator.

    As with the apply plan the scale is not in here: the emitter folds it into
    the metric components it loads, so ``fff_i = scale * g_geom_metric_i`` makes
    this the scaled energy without the plan knowing the scale exists.
    """
    dim = int(dim)
    metric = metric_matrix(dim, order=order)
    dofs = dof_symbols(dim)
    gradient = sp.Matrix([dofs[d + 1] - dofs[0] for d in range(dim)])
    energy = (gradient.T * metric * gradient)[0, 0] / 2
    temporaries, reduced = sp.cse(
        [energy], symbols=sp.numbered_symbols(temporary_prefix)
    )
    return AffineElementKernelPlan(
        dim=dim,
        temporaries=tuple(temporaries),
        outputs=tuple(reduced),
    )


@dataclass(frozen=True)
class ExpandedSimplexMetricValuePlan:
    """The closed-form 0-form loop a lowest-order simplex calls for.

    The stepped objective evaluates at ``x + alpha * h`` for each of `nsteps`
    steps, so unlike the 1-form it reads both field roles; that is why it has a
    plan of its own rather than a flag on the other one.
    """

    kernel: AffineElementKernelPlan
    scale: object

    @property
    def dim(self):
        return self.kernel.dim

    @property
    def n_shape(self):
        return self.kernel.n_shape


def expanded_simplex_metric_value_plan(
    metric,
    dim,
    n_nodes,
    n_qp,
    n_field_components,
    writes_per_shape,
    value_scale,
):
    """That plan, or ``None`` when the shape does not call for it.

    ``value_scale`` is the caller's answer to whether the energy really is
    ``scale/2 * ||grad u||^2``.  It cannot be inferred from the flux: adding a
    constant to an energy density leaves the flux untouched, so a form whose
    metric-based *gradient* is valid can still have a metric-based *value* that
    is not.  The question is asked of the density and answered before this.
    """
    if metric is None or value_scale is None or n_qp != 1:
        return None
    if dim not in (2, 3) or n_nodes != dim + 1:
        return None
    if int(n_field_components) != 1 or writes_per_shape:
        return None
    return ExpandedSimplexMetricValuePlan(
        kernel=p1_simplex_metric_value_plan(dim),
        scale=value_scale,
    )
