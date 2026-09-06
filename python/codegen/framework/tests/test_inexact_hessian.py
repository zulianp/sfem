"""The inexact Hessian action: what it costs, and where it is not inexact.

Projecting the material tangent onto the constants separates it from the
quadrature sum, leaving a per-element object and a reference tensor that is a
property of the element alone.  These tests pin the three things that make that
worth doing: the reference tensor is integrated correctly, it is small in a
specific way, and on an affine simplex the projection loses nothing at all --
which is the case the whole construction can be checked against.
"""

import itertools
import unittest

import sympy as sp

from codegen.framework.fem.reference_basis import (
    reference_basis,
    supported_elements,
)
from codegen.framework.plans.inexact_hessian import (
    inexact_hessian_plan,
    projection_is_exact,
    rank_factored_gradient_product,
    reference_gradient_product,
    staged_action,
)


#: element -> (entries, distinct values, zeros).  Recorded rather than derived:
#: these numbers are the reason the reference tensor is folded into the emitted
#: arithmetic instead of stored, and a change in them is a change in that case.
SHAPE = {
    "TRI3": (36, 3, 20),
    "TET4": (144, 3, 108),
    "QUAD4": (64, 6, 0),
    "HEX8": (576, 10, 0),
    "TET10": (900, 9, 519),
}


class ReferenceGradientProductTest(unittest.TestCase):
    def test_every_supported_element_has_one(self):
        self.assertEqual(set(supported_elements()), set(SHAPE))

    def test_the_shape_is_what_the_folding_decision_rests_on(self):
        for element, (entries, distinct, zeros) in sorted(SHAPE.items()):
            reference = reference_gradient_product(element)
            with self.subTest(element=element):
                self.assertEqual(reference.n_entries, entries)
                self.assertEqual(len(reference.distinct_values), distinct)
                self.assertEqual(reference.n_zero, zeros)

    def test_it_is_major_symmetric_everywhere(self):
        """An integral of two gradients: swapping the pairs swaps the factors."""
        for element in supported_elements():
            with self.subTest(element=element):
                self.assertTrue(reference_gradient_product(element).is_major_symmetric)

    def test_the_entries_are_rationals(self):
        """They are folded into emitted code, so they must be exact."""
        for element in supported_elements():
            for entry in reference_gradient_product(element).entries:
                with self.subTest(element=element):
                    self.assertIsInstance(sp.nsimplify(entry), sp.Rational)

    def test_the_symbolic_integration_agrees_with_quadrature(self):
        """An independent route to the same numbers.

        The tensor is integrated symbolically once at generation time; this
        integrates it again by Gauss-Legendre and compares, so that a mistake
        in the limits or the basis shows up as a number rather than as a wrong
        kernel much later.
        """
        for element in ("QUAD4", "HEX8"):
            basis = reference_basis(element)
            reference = reference_gradient_product(element)
            gradients = basis.gradients()
            points = _gauss_legendre(4)
            worst = 0.0
            for node, m, other, n in itertools.product(
                range(basis.n_nodes),
                range(basis.dim),
                range(basis.n_nodes),
                range(basis.dim),
            ):
                integrand = sp.lambdify(
                    basis.coordinates, gradients[node][m] * gradients[other][n], "math"
                )
                total = 0.0
                for combination in itertools.product(points, repeat=basis.dim):
                    weight = 1.0
                    for _coordinate, factor in combination:
                        weight *= factor
                    total += weight * integrand(
                        *[coordinate for coordinate, _factor in combination]
                    )
                worst = max(
                    worst, abs(total - float(reference.entry(node, m, other, n)))
                )
            with self.subTest(element=element):
                self.assertLess(worst, 1e-12)


class ProjectionTest(unittest.TestCase):
    def test_it_is_exact_where_the_state_does_not_vary(self):
        """The affine simplices, and only those.

        On a linear simplex the deformation gradient is constant over the cell,
        so the tangent is too, so projecting it onto the constants returns it
        unchanged.  That is the case where this kernel must reproduce the exact
        one, and it is the gate on everything the curved elements then do.
        """
        self.assertTrue(projection_is_exact("TET4"))
        self.assertTrue(projection_is_exact("TRI3"))
        for element in ("HEX8", "QUAD4", "TET10"):
            self.assertFalse(projection_is_exact(element))

    def test_an_affine_simplex_action_matches_one_point_quadrature(self):
        """The plan's action against the quadrature it replaces.

        A linear simplex has constant gradients and one quadrature point of
        weight equal to the reference measure, so the exact element action can
        be written directly.  The plan is built by symbolic integration instead
        and must land on the same expression.
        """
        plan = inexact_hessian_plan("TET4")
        basis = reference_basis("TET4")
        gradients = basis.gradients()
        tangent = sp.symbols("S0:45")
        increment = [
            [sp.Symbol("h%d_%d" % (component, node)) for node in range(plan.n_nodes)]
            for component in range(plan.dim)
        ]
        planned = plan.action(tangent, increment)

        quadrature = []
        for component in range(plan.dim):
            for node in range(plan.n_nodes):
                total = sp.Integer(0)
                for other, m, n in itertools.product(
                    range(plan.dim), repeat=3
                ):
                    gradient = sum(
                        increment[other][j] * gradients[j][m]
                        for j in range(plan.n_nodes)
                    )
                    total += (
                        tangent[plan.tangent_index(component, other, m, n)]
                        * gradient
                        * gradients[node][n]
                        * basis.measure
                    )
                quadrature.append(sp.expand(total))

        for planned_entry, quadrature_entry in zip(planned, quadrature):
            self.assertEqual(sp.simplify(planned_entry - quadrature_entry), 0)


class CompressionTest(unittest.TestCase):
    def test_folding_the_constants_beats_tabulating_them(self):
        """The measurement the design rests on.

        Storing the reference tensor's distinct values in a runtime table
        compresses the memory and destroys the arithmetic: the constants stop
        being constants and every multiply by zero survives into the emitted
        code.  Folded, TET4's action is a third of the operations.
        """
        plan = inexact_hessian_plan("TET4")
        tangent = sp.symbols("S0:45")
        increment = [
            [sp.Symbol("h%d_%d" % (component, node)) for node in range(plan.n_nodes)]
            for component in range(plan.dim)
        ]
        folded = _operations(plan.action(tangent, increment))

        table = {}
        for entry in plan.reference.entries:
            table.setdefault(sp.nsimplify(entry), sp.Symbol("Wc[%d]" % len(table)))
        tabulated = []
        for component in range(plan.dim):
            for node in range(plan.n_nodes):
                total = sp.Integer(0)
                for other, m, n in itertools.product(range(plan.dim), repeat=3):
                    weight = sum(
                        increment[other][j]
                        * table[sp.nsimplify(plan.reference.entry(j, m, node, n))]
                        for j in range(plan.n_nodes)
                    )
                    total += (
                        tangent[plan.tangent_index(component, other, m, n)] * weight
                    )
                tabulated.append(sp.expand(total))
        self.assertLess(folded, _operations(tabulated) / 3)


#: element -> rank of the reference tensor.  It is the dimension of the span of
#: the element's gradient functions, and it is where the compression is: a
#: linear simplex is rank one, which is the loperand.
RANK = {"TRI3": 1, "TET4": 1, "QUAD4": 3, "HEX8": 7, "TET10": 4}


class RankFactorisationTest(unittest.TestCase):
    def test_the_rank_is_the_span_of_the_gradient_functions(self):
        for element, rank in sorted(RANK.items()):
            factored = rank_factored_gradient_product(element)
            with self.subTest(element=element):
                self.assertEqual(factored.rank, rank)
                # Not a coincidence: Wbar is the Gram matrix of those gradients.
                basis = reference_basis(element)
                gradients = [g for row in basis.gradients() for g in row]
                monomials = sorted(
                    {
                        monomial
                        for gradient in gradients
                        for monomial in sp.Poly(
                            sp.expand(gradient), *basis.coordinates
                        ).monoms()
                    }
                )
                span = sp.Matrix(
                    [
                        [
                            sp.Poly(
                                sp.expand(gradient), *basis.coordinates
                            ).coeff_monomial(monomial)
                            for monomial in monomials
                        ]
                        for gradient in gradients
                    ]
                ).rank()
                self.assertEqual(rank, span)

    def test_the_factorisation_reproduces_the_tensor_exactly(self):
        for element in sorted(RANK):
            reference = reference_gradient_product(element)
            factored = rank_factored_gradient_product(element)
            order = factored.order
            with self.subTest(element=element):
                for row in range(order):
                    for column in range(order):
                        rebuilt = sum(
                            factored.factor_entry(a, row)
                            * factored.middle_entry(a, b)
                            * factored.factor_entry(b, column)
                            for a in range(factored.rank)
                            for b in range(factored.rank)
                        )
                        self.assertEqual(
                            sp.nsimplify(rebuilt - reference.entries[row * order + column]),
                            0,
                        )


class StagedActionTest(unittest.TestCase):
    def test_the_stages_compute_the_same_action(self):
        """Substituting the stages back must give the dense expression.

        This is the whole correctness argument for the factorisation: the
        staged form exists to keep the structure out of the emitted code, and
        it is only allowed to do that if it computes the same thing.
        """
        for element in ("TRI3", "TET4", "QUAD4"):
            plan = inexact_hessian_plan(element)
            tangent, increment, names = _symbols(plan)
            stages = staged_action(plan, tangent, increment, names)
            substitution = {}
            for stage in stages[:-1]:
                for symbol, expression in stage.assignments:
                    substitution[symbol] = expression.subs(substitution)
            with self.subTest(element=element):
                for (_name, staged), dense in zip(
                    stages[-1].assignments, plan.action(tangent, increment)
                ):
                    self.assertEqual(
                        sp.simplify(sp.expand(staged.subs(substitution)) - dense), 0
                    )

    def test_staging_is_cheaper_than_the_dense_form_everywhere(self):
        """And by how much, which is the reason to carry the stages at all.

        Expanding the stages into one expression per output and letting
        common-subexpression elimination re-discover the structure gives the
        dense count back exactly, so the saving is in keeping them.
        """
        measured = {}
        for element in sorted(RANK):
            plan = inexact_hessian_plan(element)
            tangent, increment, names = _symbols(plan)
            dense = _operations(plan.action(tangent, increment))
            staged = sum(
                _operations([expression for _symbol, expression in stage.assignments])
                for stage in staged_action(plan, tangent, increment, names)
            )
            measured[element] = (dense, staged)
            with self.subTest(element=element):
                self.assertLess(staged, dense)
        # The two elements where it matters most, pinned so a regression shows.
        self.assertLess(measured["TET10"][1] * 2, measured["TET10"][0])
        self.assertLess(measured["HEX8"][1] * 3, measured["HEX8"][0] * 2)


def _symbols(plan):
    tangent = sp.symbols("S0:%d" % plan.tangent_components)
    increment = [
        [sp.Symbol("h%d_%d" % (component, node)) for node in range(plan.n_nodes)]
        for component in range(plan.dim)
    ]
    names = [
        [sp.Symbol("out%d_%d" % (component, node)) for node in range(plan.n_nodes)]
        for component in range(plan.dim)
    ]
    return tangent, increment, names


def _gauss_legendre(n):
    import numpy as np

    nodes, weights = np.polynomial.legendre.leggauss(n)
    return [(0.5 * node + 0.5, 0.5 * weight) for node, weight in zip(nodes, weights)]


def _operations(expressions):
    temporaries, reduced = sp.cse(expressions, symbols=sp.numbered_symbols("t"))
    return sum(sp.count_ops(e) for _s, e in temporaries) + sum(
        sp.count_ops(e) for e in reduced
    )


if __name__ == "__main__":
    unittest.main()
