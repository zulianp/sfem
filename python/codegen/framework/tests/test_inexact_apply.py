"""The inexact matrix-free apply: what it costs, and where it is not inexact.

Projecting the material tangent onto the constants separates it from the
quadrature sum, leaving a per-element object and a reference tensor that is a
property of the element alone.  These tests pin the three things that make that
worth doing: the reference tensor is integrated correctly, it is small in a
specific way, and on an affine simplex the projection loses nothing at all --
which is the case the whole construction can be checked against.
"""

import dataclasses
import itertools
import unittest

import sympy as sp

from codegen.framework.fem.reference_basis import (
    reference_basis,
    supported_elements,
)
from codegen.framework.plans.inexact_apply import (
    _contraction_cost,
    contraction_ordering,
    gradient_first_action,
    inexact_apply_plan,
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
        plan = _symmetric(inexact_apply_plan("TET4"))
        basis = reference_basis("TET4")
        gradients = basis.gradients()
        tangent = sp.symbols("S0:%d" % plan.tangent_components)
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
        plan = inexact_apply_plan("TET4")
        tangent = sp.symbols("S0:%d" % plan.tangent_components)
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
            plan = _symmetric(inexact_apply_plan(element))
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

    #: element -> (dense operations, staged operations).  Recorded, because
    #: the staging is worth carrying only where the rank is small against the
    #: order, and these are the numbers that say where that is.
    #: Measured on the symmetric packing, which is what a material written as an
    #: energy has: its tangent is a Hessian.  A residual's tangent is a Jacobian
    #: and need not be symmetric, and then the store is the full square -- see
    #: `UNSYMMETRIC_COST` below.
    COST = {
        "TET4": (486, 207),
        "TRI3": (96, 52),
        "QUAD4": (238, 238),
        "HEX8": (4059, 2610),
        "TET10": (4018, 1380),
    }

    def test_the_staged_cost_is_what_was_measured(self):
        for element, (dense_expected, staged_expected) in sorted(self.COST.items()):
            plan = _symmetric(inexact_apply_plan(element))
            tangent, increment, names = _symbols(plan)
            dense = _operations(plan.action(tangent, increment))
            staged = sum(
                _operations([expression for _symbol, expression in stage.assignments])
                for stage in staged_action(plan, tangent, increment, names)
            )
            with self.subTest(element=element):
                self.assertEqual((dense, staged), (dense_expected, staged_expected))

    def test_staging_never_costs_more_and_pays_where_the_rank_is_small(self):
        """Where it pays, and the one element where it does not.

        The saving is the rank against the order.  A linear simplex is rank one
        of twelve and a tet10 rank four of thirty, and both roughly a third of
        the dense count.  QUAD4 is rank three of eight, which is not enough
        structure to beat common-subexpression elimination on the dense form:
        it comes out exactly even, and is recorded that way rather than
        presented as a win.
        """
        for element, (dense, staged) in sorted(self.COST.items()):
            with self.subTest(element=element):
                self.assertLessEqual(staged, dense)
        for element in ("TET4", "TRI3", "TET10", "HEX8"):
            dense, staged = self.COST[element]
            with self.subTest(element=element):
                self.assertLess(staged * 3, dense * 2)
        self.assertEqual(self.COST["QUAD4"][0], self.COST["QUAD4"][1])


class ContractionOrderingTest(unittest.TestCase):
    """Two ways to contract the same action, and the choice between them.

    The staged ordering carries the increment through `Wbar`'s rank
    factorisation; the gradient-first ordering contracts `Wbar` with the
    increment directly.  Which is cheaper is a property of the element -- on
    HEX8 the rank is high enough that the factorisation costs more than it
    saves -- so the plan measures rather than assumes.  What may never differ is
    the answer.
    """

    def test_both_orderings_compute_the_same_action(self):
        for element in ("TRI3", "TET4", "QUAD4", "HEX8"):
            plan = _symmetric(inexact_apply_plan(element))
            tangent, increment, names = _symbols(plan)
            dense = plan.action(tangent, increment)
            for build in (staged_action, gradient_first_action):
                stages = build(plan, tangent, increment, names)
                substitution = {}
                for stage in stages[:-1]:
                    for symbol, expression in stage.assignments:
                        substitution[symbol] = expression.subs(substitution)
                with self.subTest(element=element, ordering=build.__name__):
                    for (_name, staged), reference in zip(stages[-1].assignments, dense):
                        self.assertEqual(
                            sp.simplify(sp.expand(staged.subs(substitution)) - reference), 0
                        )

    def test_the_choice_is_the_cheaper_one(self):
        """Whatever it picks must actually be the cheaper of the two."""
        for element in ("TET4", "HEX8"):
            plan = inexact_apply_plan(element)
            tangent, increment, names = _symbols(plan)
            staged = _contraction_cost(staged_action(plan, tangent, increment, names))
            direct = _contraction_cost(gradient_first_action(plan, tangent, increment, names))
            chosen = contraction_ordering(element)
            with self.subTest(element=element, staged=staged, gradient_first=direct):
                self.assertEqual(chosen, "staged" if staged <= direct else "gradient_first")

    def test_hex8_does_not_use_the_rank_factorisation(self):
        """Pinned, because it is the one that pays for measuring.

        HEX8's `Wbar` has rank 7 of 24, high enough that the compression and
        expansion stages cost more than they save, and contracting directly is
        about 1.5x cheaper.  TET4's rank is 1 and the factorisation is decisive
        there, so the two elements must not share an ordering.
        """
        self.assertEqual(contraction_ordering("HEX8"), "gradient_first")
        self.assertEqual(contraction_ordering("TET4"), "staged")


class UnsymmetricTangentTest(unittest.TestCase):
    """The packing when the tangent is a Jacobian rather than a Hessian.

    An energy's flux is a gradient, so its tangent carries
    `A[i,j,k,l] == A[k,l,i,j]` and the store folds to half a square.  A
    residual's flux is not a gradient: the Kelvin-Voigt viscous tangent is
    genuinely unsymmetric, and folding it averaged its antisymmetric part away
    and returned a different operator.  Nothing tested the unsymmetric packing
    when that happened, which is why it took a real material to find it.
    """

    def test_the_store_is_a_full_square_without_the_symmetry(self):
        for element in ("TRI3", "TET4"):
            plan = inexact_apply_plan(element)
            order = plan.dim * plan.dim
            with self.subTest(element=element):
                self.assertFalse(plan.symmetric)
                self.assertEqual(plan.tangent_components, order * order)
                self.assertEqual(
                    _symmetric(plan).tangent_components, order * (order + 1) // 2
                )

    def test_every_index_is_distinct_without_the_symmetry(self):
        """No two tangent entries may share a slot, or one of them is lost."""
        for element in ("TRI3", "TET4"):
            plan = inexact_apply_plan(element)
            dim = plan.dim
            seen = {}
            for i, k, m, n in itertools.product(range(dim), repeat=4):
                slot = plan.tangent_index(i, k, m, n)
                self.assertNotIn(slot, seen, "%s: (%d,%d,%d,%d) collides with %s"
                                 % (element, i, k, m, n, seen.get(slot)))
                seen[slot] = (i, k, m, n)
            with self.subTest(element=element):
                self.assertEqual(len(seen), plan.tangent_components)

    def test_a_symmetric_tangent_gives_the_same_action_either_way(self):
        """The two packings agree exactly where the symmetry actually holds.

        This is what makes the symmetric packing an optimisation rather than a
        different operator: fed a tangent that really is symmetric, the folded
        store and the full square must produce the same action.
        """
        plan = inexact_apply_plan("TET4")
        folded = _symmetric(plan)
        dim = plan.dim
        entry = {}
        for i, k, m, n in itertools.product(range(dim), repeat=4):
            key = tuple(sorted(((i, n), (k, m))))
            entry.setdefault(key, sp.Symbol("A_%d%d_%d%d" % (key[0] + key[1])))
        full = [sp.Integer(0)] * plan.tangent_components
        half = [sp.Integer(0)] * folded.tangent_components
        for i, k, m, n in itertools.product(range(dim), repeat=4):
            value = entry[tuple(sorted(((i, n), (k, m))))]
            full[plan.tangent_index(i, k, m, n)] = value
            half[folded.tangent_index(i, k, m, n)] = value
        increment = [
            [sp.Symbol("h%d_%d" % (c, j)) for j in range(plan.n_nodes)]
            for c in range(dim)
        ]
        for a, b in zip(plan.action(full, increment), folded.action(half, increment)):
            self.assertEqual(sp.expand(a - b), 0)


def _symmetric(plan):
    """That plan with the major symmetry asserted.

    `inexact_apply_plan` leaves `symmetric` false, because that is the safe
    default for a plan whose flux has not been examined: storing the full square
    for a symmetric tangent costs memory, storing half of an unsymmetric one
    silently returns a different operator.  Tests that measure the symmetric
    packing have to ask for it.
    """
    return dataclasses.replace(plan, symmetric=True)


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


class AgainstTheExactActionTest(unittest.TestCase):
    """The only test that can catch an index convention being wrong.

    Everything else here compares the construction against itself: the staged
    form against the dense form, the factorisation against the tensor.  Those
    all pass with the tangent packed on the wrong symmetry, because both sides
    share it.  This integrates the element action directly on a concrete
    tetrahedron and compares, which is what found `S[i,k,m,n] == S[k,i,n,m]`
    after `(i,k)` against `(m,n)` had looked plausible and been wrong.

    It also settles the claim the whole design rests on: on an affine simplex
    the projection loses nothing *even for a nonlinear material*, because the
    deformation gradient does not vary over the cell.
    """

    MATERIALS = (
        # (material, whether its tangent depends on the state)
        ("linear_elasticity", False),
        ("neohookean_ogden", True),
    )

    def test_the_projected_action_is_the_exact_one_on_a_tetrahedron(self):
        for material, nonlinear in self.MATERIALS:
            with self.subTest(material=material):
                dense, staged = self._compare(material, nonlinear)
                self.assertEqual(dense, 0)
                self.assertEqual(staged, 0)

    def _compare(self, material, nonlinear):
        flux_form, dim = self._flux_form(material)
        gradient, flux = list(flux_form.gradient), list(flux_form.flux)
        parameters = {
            sp.Symbol("mu"): sp.Rational(7, 3),
            sp.Symbol("lmbda"): sp.Rational(11, 5),
        }
        # A deformation gradient near the identity: a hyperelastic tangent is
        # only defined where its determinant is positive.
        state = {}
        for index, symbol in enumerate(gradient):
            row, column = divmod(index, dim)
            state[symbol] = (sp.Integer(1) if row == column else sp.Integer(0)) + (
                sp.Rational(index + 1, 53) if nonlinear else sp.Integer(0)
            )
        tangent = {
            (i, j, k, l): sp.nsimplify(
                sp.diff(flux[i * dim + j], gradient[k * dim + l])
                .subs(parameters)
                .subs(state)
            )
            for i, j, k, l in itertools.product(range(dim), repeat=4)
        }

        vertices = [
            sp.Matrix([0, 0, 0]),
            sp.Matrix([sp.Rational(3, 2), 0, 0]),
            sp.Matrix([sp.Rational(1, 4), sp.Rational(5, 4), 0]),
            sp.Matrix([sp.Rational(1, 3), sp.Rational(1, 5), sp.Rational(7, 6)]),
        ]
        jacobian = sp.Matrix(
            3, 3, lambda r, c: vertices[c + 1][r] - vertices[0][r]
        )
        determinant = jacobian.det()
        inverse = jacobian.inv()
        basis = reference_basis("TET4")
        gradients = basis.gradients()
        volume = determinant * basis.measure
        increment = [
            [sp.Rational((k + 1) * (j + 2), 7 + k + j) for j in range(4)]
            for k in range(dim)
        ]

        physical = sp.zeros(dim, dim)
        for component in range(dim):
            for axis in range(dim):
                physical[component, axis] = sum(
                    increment[component][node]
                    * sum(gradients[node][m] * inverse[m, axis] for m in range(dim))
                    for node in range(4)
                )
        exact = sp.zeros(dim, 4)
        for i in range(dim):
            for p in range(4):
                test = [
                    sum(gradients[p][m] * inverse[m, axis] for m in range(dim))
                    for axis in range(dim)
                ]
                exact[i, p] = (
                    sum(
                        tangent[(i, j, k, l)] * physical[k, l] * test[j]
                        for j, k, l in itertools.product(range(dim), repeat=3)
                    )
                    * volume
                )

        plan = inexact_apply_plan("TET4")
        packed = [0] * plan.tangent_components
        for i, k, m, n in itertools.product(range(dim), repeat=4):
            packed[plan.tangent_index(i, k, m, n)] = (
                sum(
                    tangent[(i, j, k, l)] * inverse[n, j] * inverse[m, l]
                    for j, l in itertools.product(range(dim), repeat=2)
                )
                * determinant
            )

        dense = plan.action(packed, increment)
        names = [
            [sp.Symbol("o%d_%d" % (i, p)) for p in range(4)] for i in range(dim)
        ]
        stages = staged_action(plan, packed, increment, names)
        substitution = {}
        for stage in stages[:-1]:
            for symbol, expression in stage.assignments:
                substitution[symbol] = expression.subs(substitution)
        staged = [
            expression.subs(substitution) for _name, expression in stages[-1].assignments
        ]
        worst_dense = max(
            abs(sp.nsimplify(dense[i * 4 + p] - exact[i, p]))
            for i in range(dim)
            for p in range(4)
        )
        worst_staged = max(
            abs(sp.nsimplify(staged[i * 4 + p] - exact[i, p]))
            for i in range(dim)
            for p in range(4)
        )
        return worst_dense, worst_staged

    @staticmethod
    def _flux_form(name):
        import sys, os

        materials = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials"
        )
        if materials not in sys.path:
            sys.path.insert(0, materials)
        import importlib

        from codegen.framework.symbolic.weak_forms import (
            flux_form_from_energy,
            sfem_soa_weak_form,
        )

        module = importlib.import_module(name)
        system = list(module.systems.systems)[-1]
        collection = system.form_collection(list(system.equations)[0])
        weak_form = sfem_soa_weak_form(
            collection.forms[0].expression,
            sp.Matrix(
                len(collection.variables) // system.dim,
                system.dim,
                list(collection.variables),
            ),
        )
        return flux_form_from_energy(weak_form), system.dim


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
