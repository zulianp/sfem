"""The evaluation strategy an element gets, stated once.

Three defaults govern how a form is evaluated, and each belongs to the element:
tensor-product elements are sum-factorised, lowest-order simplices are evaluated
in expanded closed form with no quadrature data, higher-order simplices have
their own rules.

The point of the plan this pins is what it does *not* take.  `evaluation_strategy`
receives an element and nothing else -- no dependencies, no form, no `FormKind` --
because a strategy cannot track the formulation if the formulation is not an
argument.  Today it does track it: `tests/test_contraction_mode.py` records that
coupling, and this is the plan that replaces it.
"""

import inspect
import unittest

from codegen.framework.fem.element_family import ElementFamily, element_family
from codegen.framework.plans.evaluation_strategy import (
    ElementEvaluationPlan,
    EvaluationStrategy,
    element_evaluation_plan,
    evaluation_strategy,
)


class EvaluationStrategyPlanTest(unittest.TestCase):
    maxDiff = None

    def test_each_family_gets_the_strategy_its_rule_names(self):
        for element, family, strategy in (
            ("HEX8", ElementFamily.TENSOR_PRODUCT, EvaluationStrategy.SUM_FACTORIZED),
            ("HEX27", ElementFamily.TENSOR_PRODUCT, EvaluationStrategy.SUM_FACTORIZED),
            ("QUAD4", ElementFamily.TENSOR_PRODUCT, EvaluationStrategy.SUM_FACTORIZED),
            ("PROTEUS_HEX8", ElementFamily.TENSOR_PRODUCT, EvaluationStrategy.SUM_FACTORIZED),
            ("TET4", ElementFamily.SIMPLEX_LOWEST, EvaluationStrategy.EXPANDED),
            ("TRI3", ElementFamily.SIMPLEX_LOWEST, EvaluationStrategy.EXPANDED),
            ("TET10", ElementFamily.SIMPLEX_HIGHER, EvaluationStrategy.QUADRATURE),
            ("TRI6", ElementFamily.SIMPLEX_HIGHER, EvaluationStrategy.QUADRATURE),
        ):
            with self.subTest(element=element):
                self.assertIs(element_family(element), family)
                self.assertIs(evaluation_strategy(element), strategy)

    def test_a_mixed_order_pair_is_not_claimed_by_any_rule(self):
        """Taylor-Hood fields live on different spaces; the rules are silent."""
        for element in ("HEX27_HEX8", "TET10_TET4", "TRI6_TRI3"):
            with self.subTest(element=element):
                self.assertIs(element_family(element), ElementFamily.MIXED)
                self.assertIs(
                    evaluation_strategy(element), EvaluationStrategy.QUADRATURE
                )

    def test_the_strategy_depends_on_the_element_and_nothing_else(self):
        """The signature is the guarantee, so the signature is what is pinned."""
        for function in (evaluation_strategy, element_evaluation_plan):
            with self.subTest(function=function.__name__):
                parameters = list(
                    inspect.signature(function).parameters
                )
                self.assertEqual(
                    parameters,
                    ["element_type"],
                    "%s must take the element and nothing else; a strategy "
                    "cannot stop tracking the formulation while the "
                    "formulation is still reachable from here"
                    % function.__name__,
                )

    def test_only_the_expanded_family_skips_quadrature_and_basis_data(self):
        """The observable the conformance report counts."""
        expanded = element_evaluation_plan("TET4")
        self.assertFalse(expanded.emits_quadrature_loop)
        self.assertFalse(expanded.needs_reference_basis_data)

        factorised = element_evaluation_plan("HEX8")
        self.assertFalse(factorised.emits_quadrature_loop)
        self.assertTrue(
            factorised.needs_reference_basis_data,
            "sum factorization needs the one-dimensional tables",
        )

        general = element_evaluation_plan("TET10")
        self.assertTrue(general.emits_quadrature_loop)
        self.assertTrue(general.needs_reference_basis_data)

    def test_the_report_and_the_plan_share_one_taxonomy(self):
        """A report that classified elements differently would measure fiction."""
        from codegen.framework.tools import evaluation_strategy as report

        for element in ("hex8", "tet4", "tri3", "tet10", "proteus_hex8"):
            with self.subTest(element=element):
                self.assertEqual(
                    report.element_family(element),
                    element_family(element).value,
                )


if __name__ == "__main__":
    unittest.main()
