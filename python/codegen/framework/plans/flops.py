"""What one element costs, derived from the loops the generator emits.

Every generated operator publishes a ``KernelDiagnostics`` record, and every
consumer of a timing reads

    total_flops = nelements * (n_qp * flops_per_qp_lane + mesh_flops_per_element)

That number rescales every GFLOP/s and every arithmetic intensity reported for
these kernels, so it is not a decoration: getting it wrong misdirects
optimisation work.

It was wrong.  ``flops_per_qp_lane`` is the cost of the *material* expression
graph -- the constitutive evaluation at one point -- and the mesh term was
supposed to carry everything else: the geometry, the interpolation of the field
onto the point, and the contraction back onto the test functions.  It was
computed by ``_tensor_product_mesh_extra_flops_per_element`` in
``emitters/energy_codegen.py``, whose first statement was

    if str(basis_family) != "tensor_product":
        return 0, 0

so on every simplex element the mesh term was exactly zero and ``total_flops``
reported the material cost alone.  Measured on the metric-specialised TET4
Laplacian, whose element body is straight-line and therefore countable
exactly: the model said 3 operations and the emitted kernel contains 30.  On
TRI3, 2 against 15.  ``tools/flops_audit.py`` is that measurement.

Two things were wrong at once, and they want different remedies.

**The model lived in the emitter.**  Emission printed a number it had derived
itself, from an arithmetic table of its own, gated on a fact -- the basis
family -- that the plan layer already owns.  So the model moves here, and the
emitter prints what it is handed.

**The composition was written once per family instead of once.**  The
tensor-product branch composed the element cost out of five stages: the field
gradient, the physical-gradient transform, the operand transform back to the
reference, the test-gradient contraction, and (for an isoparametric mesh) the
Jacobian with its adjugate and determinant.  That composition is not a property
of sum factorization; it is what every one of these kernels does.  What differs
between families is only how much each stage costs, because sum factorization
contracts dimension by dimension where the direct path contracts over all the
shape functions at once.  So the composition is written once below and the
per-stage counts are chosen by the evaluation strategy.

A note on what these numbers are.  For the ``EXPANDED`` strategy the count is
*exact*: the arithmetic is a sympy expression list on the plan, and counting it
counts precisely what is printed.  For the other two it is a structural model
-- the operation counts implied by the loop bounds -- which is what the
diagnostics can honestly claim without a full cost interpreter over the
emitted C.  It is accurate to the shape of the loops, and the alternative it
replaces was zero.
"""

from dataclasses import dataclass

from codegen.framework.plans.evaluation_strategy import (
    EvaluationStrategy,
    evaluation_strategy,
)
from codegen.framework.plans.scheduling import ExpressionCost, _op_counts


def expression_flops(expressions):
    """The weighted operation count of a list of sympy expressions.

    The same weights ``ExpressionCost.flops`` applies, so a count taken from an
    expression list and one taken from an expression graph are on one scale.
    """
    totals = [0] * 8
    for expression in expressions:
        for index, value in enumerate(_op_counts(expression)):
            totals[index] += value
    adds, muls, divs, sqrts, pows, exps, logs, trigs = totals
    return ExpressionCost(
        adds=adds,
        muls=muls,
        divs=divs,
        sqrts=sqrts,
        pows=pows,
        exps=exps,
        logs=logs,
        trigs=trigs,
    ).flops


# ---------------------------------------------------------------------------
# The stages, once each.
#
# Every one of these counts a construct the emitters print, and says which.
# ---------------------------------------------------------------------------


def direct_gradient_flops(dim, n_qp, n_shape):
    """One component's reference gradient at every point, contracted directly.

    The loop the general body prints is ``for q`` over ``for shape`` over
    ``for d``, accumulating ``value * grad_ref[...]``: one multiply and one add
    per shape function, per direction, per point.  The same shape serves the
    field gradient and the test-function contraction, which are transposes of
    each other.
    """
    return 2 * int(dim) * int(n_qp) * int(n_shape)


def sum_factorized_gradient_flops(dim, n_qp_1d, n_shape_1d):
    """The same gradient, contracted dimension by dimension.

    Transcribed from the counts ``emitters/energy_codegen.py`` carried for the
    tensor-product path, which follow the three contraction sweeps
    ``tensor_product_kernels.hpp`` performs: each sweep replaces one index of
    the tensor and costs a multiply-add per output entry, with the leading
    sweep also loading and scaling.
    """
    dim = int(dim)
    q = int(n_qp_1d)
    s = int(n_shape_1d)
    if dim == 2:
        return 4 * q * s * s + 6 * q * q * s
    if dim == 3:
        return 4 * q * s * s * s + 6 * q * q * s * s + 6 * q * q * q * s
    return 0


def sum_factorized_test_gradient_flops(dim, n_qp_1d, n_shape_1d):
    """The transpose sweeps, which visit the same tensors in the other order."""
    dim = int(dim)
    q = int(n_qp_1d)
    s = int(n_shape_1d)
    if dim == 2:
        return 6 * q * q * s + 5 * q * s * s
    if dim == 3:
        return 6 * q * q * q * s + 6 * q * q * s * s + 5 * q * s * s * s
    return 0


#: One point's adjugate and determinant, by dimension.
#:
#: These count the bodies of ``geometry_jacobian_adjugate_and_determinant_2``
#: and ``_3`` in ``frontend/ops/generated/geometry_kernels.hpp``, which are
#: hand-written templates rather than expressions the generator builds -- so
#: the count is declared here and pinned against the template by
#: ``tests/test_flops_model_matches_the_ir.py``, which counts them and fails if
#: either drifts.
#:
#: The two-dimensional entry read 11 and the template contains 3: a 2x2
#: adjugate is four sign flips and copies, and only the determinant does any
#: arithmetic.  The three-dimensional entry was right.
ADJUGATE_AND_DETERMINANT_FLOPS_PER_QP = {2: 3, 3: 41}


def adjugate_and_determinant_flops_per_qp(dim):
    """One point's adjugate and determinant, from ``geometry_kernels.hpp``."""
    return ADJUGATE_AND_DETERMINANT_FLOPS_PER_QP.get(int(dim), 0)


def physical_gradient_transform_flops_per_qp(dim):
    """Mapping one reference gradient to physical: ``adj * gref / det``."""
    dim = int(dim)
    if dim <= 0:
        return 0
    return 1 + dim * dim * (2 * dim)


def operand_transform_flops_per_qp(dim):
    """Mapping the flux back to the reference frame for the contraction."""
    dim = int(dim)
    if dim <= 0:
        return 0
    return dim * dim * (2 * dim)


def objective_weight_flops_per_qp():
    """Weighting one point's energy density by ``qw * det`` and accumulating."""
    return 3


#: Which forms contract onto test functions and which reduce to a scalar.  A
#: table rather than a chain of `if form_name == ...`, for the same reason the
#: body tables in the emitter are: the distinction is a property of the form's
#: order, and the model looks it up.
_CONTRACTS_ONTO_TEST_FUNCTIONS = {
    "objective": False,
    "value": False,
    "gradient": True,
    "apply": True,
    "residual": True,
    "jacobian_action": True,
}


def contracts_onto_test_functions(form_name):
    """Whether this form's result is contracted back onto the test functions.

    A 0-form reduces its integrand to one number per element and pays a
    weighting instead; everything else carries the result back to the
    reference frame and contracts it, which is the larger of the two.  An
    unrecognised name is assumed to contract, because that is what all but the
    0-forms do and because understating a cost is the failure this model was
    built to stop.
    """
    return _CONTRACTS_ONTO_TEST_FUNCTIONS.get(str(form_name), True)


@dataclass(frozen=True)
class ElementFlopsPlan:
    """One element's cost, split the way the diagnostics record splits it.

    ``material_flops_per_qp`` is the constitutive evaluation and is reported as
    ``flops_per_qp_lane``; the two mesh terms are everything else, so that
    ``n_qp * material + mesh`` is the element total for that geometry mode.
    """

    n_qp: int
    material_flops_per_qp: int
    affine_flops_per_element: int
    isoparametric_flops_per_element: int
    #: True when the affine total was counted from an expression list rather
    #: than modelled from the loop bounds.
    affine_is_exact: bool = False

    @property
    def affine_mesh_flops_per_element(self):
        return max(
            0,
            self.affine_flops_per_element - self.n_qp * self.material_flops_per_qp,
        )

    @property
    def isoparametric_mesh_flops_per_element(self):
        return max(
            0,
            self.isoparametric_flops_per_element
            - self.n_qp * self.material_flops_per_qp,
        )


def closed_form_element_flops(expanded_plan, scale_is_unit):
    """The exact cost of a closed-form simplex body, from its own expressions.

    ``emitters/energy_codegen.py:_expanded_simplex_metric_body`` prints, in
    order: one scaling of each metric component when the scale is not one, the
    plan's temporaries, its outputs, and one accumulation per shape function
    into the element vector.  Counting the plan counts the body.
    """
    kernel = expanded_plan.kernel
    expressions = [expression for _symbol, expression in kernel.temporaries]
    expressions.extend(kernel.outputs)
    flops = expression_flops(expressions)
    if not scale_is_unit:
        from codegen.framework.plans.form_transformations import (
            symmetric_metric_component_count,
        )

        flops += symmetric_metric_component_count(kernel.dim)
    return flops + kernel.n_shape


def _stage_gradient_flops(strategy, dim, n_qp, n_shape, quadrature_rule):
    """One component's gradient over the whole element, for this strategy."""
    if strategy is EvaluationStrategy.SUM_FACTORIZED:
        return sum_factorized_gradient_flops(
            dim,
            quadrature_rule.tensor_product_n_qp_1d,
            quadrature_rule.tensor_product_n_shape_1d,
        )
    return direct_gradient_flops(dim, n_qp, n_shape)


def _stage_test_gradient_flops(strategy, dim, n_qp, n_shape, quadrature_rule):
    """One component's contraction onto the test functions, for this strategy."""
    if strategy is EvaluationStrategy.SUM_FACTORIZED:
        return sum_factorized_test_gradient_flops(
            dim,
            quadrature_rule.tensor_product_n_qp_1d,
            quadrature_rule.tensor_product_n_shape_1d,
        )
    return direct_gradient_flops(dim, n_qp, n_shape)


def local_element_flops(
    form_name,
    dim,
    n_qp,
    n_shape,
    n_field_components,
    quadrature_rule,
    strategy,
    material_flops_per_qp,
):
    """The element body's cost, excluding the geometry it is handed.

    The composition every one of these kernels performs, written once: bring
    each field component onto the quadrature points, map its gradient to the
    physical frame, evaluate the material there, and -- for a form that has
    test functions -- map the result back and contract it onto them.
    """
    n_qp = int(n_qp)
    field = _stage_gradient_flops(strategy, dim, n_qp, n_shape, quadrature_rule)
    total = int(n_field_components) * field
    total += n_qp * physical_gradient_transform_flops_per_qp(dim)
    total += n_qp * int(material_flops_per_qp)
    if contracts_onto_test_functions(form_name):
        total += n_qp * operand_transform_flops_per_qp(dim)
        total += int(n_field_components) * _stage_test_gradient_flops(
            strategy, dim, n_qp, n_shape, quadrature_rule
        )
    else:
        total += n_qp * objective_weight_flops_per_qp()
    return total


def isoparametric_geometry_flops(dim, n_qp, n_shape, quadrature_rule, strategy):
    """Building the Jacobian, adjugate and determinant from the coordinates.

    An affine kernel is handed these precomputed, one set per element, and pays
    nothing for them inside the loop; an isoparametric one builds them at every
    point out of the same gradient contraction the field uses, once per
    coordinate direction.
    """
    return int(dim) * _stage_gradient_flops(
        strategy, dim, n_qp, n_shape, quadrature_rule
    ) + int(n_qp) * adjugate_and_determinant_flops_per_qp(dim)


def element_flops_plan(
    form_name,
    element_type,
    dim,
    n_qp,
    n_shape,
    n_field_components,
    quadrature_rule,
    material_flops_per_qp,
    expanded_plan=None,
    scale_is_unit=True,
):
    """One element's cost in both geometry modes.

    ``expanded_plan`` is the closed-form simplex body when the affine kernel is
    that body rather than the general one; where it is given, the affine total
    is exact.
    """
    strategy = evaluation_strategy(element_type)
    material_flops_per_qp = int(material_flops_per_qp)
    local = local_element_flops(
        form_name,
        dim,
        n_qp,
        n_shape,
        n_field_components,
        quadrature_rule,
        strategy,
        material_flops_per_qp,
    )
    geometry = isoparametric_geometry_flops(
        dim, n_qp, n_shape, quadrature_rule, strategy
    )
    affine = local
    affine_is_exact = False
    if expanded_plan is not None:
        affine = closed_form_element_flops(expanded_plan, scale_is_unit)
        affine_is_exact = True
    return ElementFlopsPlan(
        n_qp=int(n_qp),
        material_flops_per_qp=material_flops_per_qp,
        affine_flops_per_element=affine,
        isoparametric_flops_per_element=local + geometry,
        affine_is_exact=affine_is_exact,
    )
