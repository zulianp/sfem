"""How a form is evaluated, decided by the element and by nothing else.

Three defaults, each a property of the element:

    tensor-product          sum factorization, always -- there is no decision
                            to model, and both existing template families in
                            ``tensor_product_kernels.hpp`` implement it
    simplex, lowest order   the CSE expanded for a compact physical gradient,
                            and no quadrature-point data generated at all
    simplex, higher order   rules of its own

The signature below takes the element and returns the strategy.  It takes no
dependencies, no form and no ``FormKind`` -- that is the entire point.  A
strategy cannot track the formulation if the formulation is not an argument,
and today it does track it: the energy front end reaches sum factorization
through ``tensor_test`` and the residual front end through
``tensor_evaluate``/``tensor_integrate``, with nothing stating that either is a
consequence of the element.

``EXPANDED`` is the one the generator does not honour.  Every material emits
quadrature loops on TET4 and TRI3 -- twenty such elements across the tree --
where the basis gradients are constant and a one-trip loop over a single point
is pure structure.  The compact form exists only as hand-written C transcribed
into the residual emitter for the Laplacian.

A note that belongs next to this enum, because it is the thing most likely to
be got wrong by someone implementing ``EXPANDED``.  "Expanded" refers to the
loop and quadrature structure, not to the algebra.  Measured on the TET4
Laplacian element vector: contracting the metric and running CSE on the
*factored* expression gives 3 temporaries and 45 operations, while calling
``sp.expand`` first gives 15 temporaries and 50 operations -- worse than the
hand-written kernel it would replace, which has 6 and 47.  Expanding the
algebra defeats the rule.
"""

from dataclasses import dataclass
from enum import Enum

from codegen.framework.fem.element_family import ElementFamily, element_family


class EvaluationStrategy(Enum):
    """The evaluation a form gets on a given element."""

    #: Contract through one-dimensional operators, dimension by dimension.
    SUM_FACTORIZED = "sum_factorized"

    #: Evaluate in closed form over the element: no quadrature loop, no
    #: per-point geometry, no reference-basis tables.
    EXPANDED = "expanded"

    #: The general path: a quadrature loop with per-point basis and geometry.
    QUADRATURE = "quadrature"


#: The rule, as a table rather than as a chain of conditions.
STRATEGY_BY_FAMILY = {
    ElementFamily.TENSOR_PRODUCT: EvaluationStrategy.SUM_FACTORIZED,
    ElementFamily.SIMPLEX_LOWEST: EvaluationStrategy.EXPANDED,
    ElementFamily.SIMPLEX_HIGHER: EvaluationStrategy.QUADRATURE,
    ElementFamily.MIXED: EvaluationStrategy.QUADRATURE,
}


@dataclass(frozen=True)
class ElementEvaluationPlan:
    """What the strategy implies for the shape of an emitted kernel."""

    element_type: str
    family: ElementFamily
    strategy: EvaluationStrategy

    @property
    def emits_quadrature_loop(self):
        """Whether a kernel for this element loops over quadrature points.

        False for the expanded family, which is the observable this rule is
        about: ``tools/evaluation_strategy.py`` counts exactly this.
        """
        return self.strategy is EvaluationStrategy.QUADRATURE

    @property
    def needs_reference_basis_data(self):
        """Whether per-point shape and gradient tables must be generated.

        A sum-factorised kernel needs the one-dimensional tables; an expanded
        one needs nothing, because the gradients are constants folded into the
        arithmetic.
        """
        return self.strategy is not EvaluationStrategy.EXPANDED


def evaluation_strategy(element_type):
    """The strategy this element's family calls for."""
    return STRATEGY_BY_FAMILY[element_family(element_type)]


def element_evaluation_plan(element_type):
    """The strategy and what it implies, for one element."""
    family = element_family(element_type)
    return ElementEvaluationPlan(
        element_type=str(element_type).upper(),
        family=family,
        strategy=STRATEGY_BY_FAMILY[family],
    )


#: The C scope a kernel body opens, per strategy.  A table rather than a
#: conditional, and in the plan rather than in the emitter: emission printing
#: `if strategy is EXPANDED` would be emission deciding, which is the thing the
#: printer discipline is there to stop.  ``%(indent)s`` is filled by the caller.
QUADRATURE_SCOPE_LINES = {
    EvaluationStrategy.EXPANDED: (
        "%(indent)s{",
        "%(indent)s    const int q = 0;  // %(element)s evaluates in closed form",
    ),
    EvaluationStrategy.SUM_FACTORIZED: (
        "%(indent)sfor (int q = 0; q < N_QP; ++q) {",
    ),
    EvaluationStrategy.QUADRATURE: (
        "%(indent)sfor (int q = 0; q < N_QP; ++q) {",
    ),
}


def quadrature_scope_lines(element_type, indent=""):
    """The scope a kernel body opens over quadrature, for this element.

    A lowest-order simplex has one quadrature point and constant basis
    gradients, so the loop has one trip and collapses to the point itself.
    ``const int q = 0`` rather than substituting zero throughout: the bodies
    index reference tables as ``[q * N_SHAPE + shape]`` at a dozen sites, and
    the compiler folds that where rewriting each site would not be worth the
    churn.  What leaves the emitted source is the loop.
    """
    strategy = evaluation_strategy(element_type)
    substitution = {"indent": indent, "element": str(element_type).upper()}
    return [line % substitution for line in QUADRATURE_SCOPE_LINES[strategy]]
