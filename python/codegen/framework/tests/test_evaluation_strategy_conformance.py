"""The element's family decides how the generated tree evaluates it.

`PRESCRIBED_ARCHITECTURE.md` states the invariant -- evaluation strategy
follows the element, a tensor-product element is sum factorized and a
lowest-order simplex evaluates in closed form with no quadrature loop and no
per-point data -- and `plans/evaluation_strategy.py` decides it.  Nothing made
the generated tree obey it, which is how it drifted to twelve departures
unnoticed: every lowest-order simplex kernel still opened a one-trip loop over
its single quadrature point.

`tools/evaluation_strategy.py` is the measurement, and this is the ratchet on
it.  The budget is zero and it is two-sided: a lowest-order simplex that grows
a quadrature loop fails here, which is the point, and so does a departure
counted after the gap has been closed.

What the rule does *not* cover is worth stating, because assuming it did cost a
round of work.  A facet is not a small cell.  A lowest-order simplex is
loop-free because its volume integrand is built from basis gradients that are
constant over the cell, so a single point is exact and
`sfem_element_quadrature_rule("TET4")` carries exactly one.  A facet load
vector integrates the shape function itself -- and, for `neumann_general`, a
traction that varies in space -- so `boundary_codegen` asks for degree
`2 * order` and a flat TRISHELL3 facet genuinely carries three points.  That
loop runs three times; collapsing it to `q = 0` keeps a third of the traction.
Surface integrals are therefore outside both rules rather than departures from
them, which is the same call the report already made for sum factorization.
"""

import os
import unittest

from codegen.framework.tools.evaluation_strategy import survey, violations


def _generated_tree():
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "frontend",
            "ops",
            "generated",
        )
    )


#: Departures from the element's default evaluation strategy, in the shipped
#: tree.
#:
#: 12 when the report was first run: laplace, linear_elasticity and
#: neohookean_ogden one apiece on tet4 and on tri3, mooney_rivlin_kelvin_voigt
#: four apiece, and neumann / neumann_general one apiece.
#:
#: 12 -> 4 when the expanded strategy reached the volume kernels and the patch
#: merit kernel stopped hard-coding its own loop.
#:
#: 4 -> 0 when the four remaining -- all of them facet loads -- were recognised
#: as outside the rule rather than departures from it.  See the module
#: docstring: their rules genuinely carry more than one point.
#:
#: A floor, not a target that has been met: it may only go down.
DEPARTURE_BUDGET = 0


class EvaluationStrategyConformance(unittest.TestCase):
    def test_the_generated_tree_evaluates_each_element_by_its_family(self):
        found = violations(survey(_generated_tree()))
        self.assertLessEqual(
            len(found),
            DEPARTURE_BUDGET,
            "the generated tree departs from the element's default evaluation "
            "strategy in %d places, above the budget of %d:\n    %s"
            % (
                len(found),
                DEPARTURE_BUDGET,
                "\n    ".join("%s: %s" % entry for entry in found),
            ),
        )

    def test_the_budget_records_what_the_tree_reached(self):
        found = violations(survey(_generated_tree()))
        self.assertEqual(
            len(found),
            DEPARTURE_BUDGET,
            "the tree now has %d departures against a budget of %d -- lower "
            "the budget to record the improvement, so it cannot be spent "
            "again" % (len(found), DEPARTURE_BUDGET),
        )

    def test_a_lowest_order_simplex_that_grew_a_loop_is_caught(self):
        # The ratchet is only worth anything if it fails when the gap reopens,
        # and the gap reopened silently once already.
        rows = [
            {
                "material": "invented",
                "element": "tet4",
                "family": "simplex-lowest",
                "sum_factorised": True,
                "quadrature_loops": 1,
                "has_volume_kernels": True,
            }
        ]
        self.assertEqual(
            [kind for kind, _ in violations(rows)],
            ["lowest-order simplex generating quadrature data"],
        )


if __name__ == "__main__":
    unittest.main()
