"""The stored tangent: its packing, and the action it reproduces.

The inexact apply stores the material tangent at every point of the rule the
exact apply integrates with (`plans/point_tangent.py`), grouped the way the
element allows (`plans/inexact_element.py`): point values, P2 moments on TET10,
the fold on the hexahedra.  Every one of those is claimed to be exact.  These
tests check the claims where they can fail: against the element action
integrated directly, with an index convention that shares nothing with the
packing under test.
"""

import importlib
import itertools
import os
import sys
import unittest

import numpy as np
import sympy as sp

from codegen.framework.fem.reference import sfem_element_quadrature_rule
from codegen.framework.plans.inexact_element import (
    inexact_element_plan,
    packed_slot,
    symmetric_components,
)
from codegen.framework.plans.point_tangent import (
    PointTangentPlan,
    flux_tangent_is_symmetric,
    geometry_symbols,
    physical_field_gradient_definitions,
    point_tangent,
    reference_field_gradient_symbols,
)


MATERIALS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials")


def _energy_flux_form(name):
    """The 3D flux form of a single-energy material, straight from its system."""
    if MATERIALS not in sys.path:
        sys.path.insert(0, MATERIALS)
    from codegen.framework.forms.weak_forms import flux_form_from_energy, sfem_soa_weak_form

    module = importlib.import_module(name)
    system = list(module.systems.systems)[-1]
    collection = system.form_collection(list(system.equations)[0])
    weak_form = sfem_soa_weak_form(
        collection.forms[0].expression,
        sp.Matrix(len(collection.variables) // system.dim, system.dim, list(collection.variables)),
    )
    return flux_form_from_energy(weak_form), weak_form.is_deformation_gradient


class PackingTest(unittest.TestCase):
    def test_the_symmetric_tangent_is_45_numbers_and_the_full_one_81(self):
        self.assertEqual(PointTangentPlan(3, symmetric=True).tangent_components, 45)
        self.assertEqual(PointTangentPlan(3, symmetric=False).tangent_components, 81)
        self.assertEqual(symmetric_components(3), 45)

    def test_the_symmetry_pairs_output_and_test_against_increment(self):
        """`S[i,k,m,n] == S[k,i,n,m]`: `(i,n)` swaps with `(k,m)`, never `(i,k)` with `(m,n)`."""
        plan = PointTangentPlan(3, symmetric=True)
        for i, k, m, n in itertools.product(range(3), repeat=4):
            self.assertEqual(plan.tangent_index(i, k, m, n), plan.tangent_index(k, i, n, m))

    def test_every_index_is_distinct_without_the_symmetry(self):
        plan = PointTangentPlan(3, symmetric=False)
        indices = {plan.tangent_index(i, k, m, n) for i, k, m, n in itertools.product(range(3), repeat=4)}
        self.assertEqual(len(indices), 81)

    def test_the_element_tables_use_the_same_packing(self):
        """The contraction (`packed_slot`) and the material (`tangent_index`) agree."""
        for symmetric in (True, False):
            plan = PointTangentPlan(3, symmetric=symmetric)
            for i, k, m, n in itertools.product(range(3), repeat=4):
                self.assertEqual(
                    plan.tangent_index(i, k, m, n), packed_slot(i * 3 + n, k * 3 + m, 9, symmetric)
                )


class SymmetryTest(unittest.TestCase):
    def test_an_energy_tangent_is_symmetric(self):
        flux_form, _ = _energy_flux_form("neohookean_ogden")
        self.assertTrue(flux_tangent_is_symmetric(flux_form))


class AgainstTheExactActionTest(unittest.TestCase):
    """The stored action against the element action integrated directly.

    On a curved element and a state that varies over it, so neither the
    geometry nor the tangent is constant: the stored tangent is pulled back per
    point, and that is what makes the moments (TET10) and the fold (the
    hexahedra) exact everywhere, not only on affine cells.
    """

    CASES = (("TET4", 1), ("TET10", 4), ("HEX8", 2), ("PROTEUS_HEX27", 4))

    def test_the_stored_action_is_the_exact_one(self):
        flux_form, is_deformation_gradient = _energy_flux_form("neohookean_ogden")
        point = self._point_function(flux_form, is_deformation_gradient)
        exact_tangent = self._exact_tangent_function(flux_form)
        shift = np.eye(3) if is_deformation_gradient else np.zeros((3, 3))
        for element, order in self.CASES:
            for fold in (False, True):
                with self.subTest(element=element, fold=fold):
                    plan = inexact_element_plan(element, sfem_element_quadrature_rule(element, order))
                    if fold and not plan.foldable:
                        continue
                    stored, exact = self._actions(plan, point, exact_tangent, fold, shift)
                    scale = np.max(np.abs(exact))
                    self.assertLess(np.max(np.abs(stored - exact)), 1e-11 * scale)

    @staticmethod
    def _point_function(flux_form, is_deformation_gradient):
        plan = PointTangentPlan(3, symmetric=True)
        adjugate, determinant = geometry_symbols(3)
        tangent, gradient, _previous = point_tangent(plan, flux_form, adjugate, determinant, is_deformation_gradient)
        reference = reference_field_gradient_symbols(3, "gu")
        definitions = dict(physical_field_gradient_definitions(reference, gradient, adjugate, determinant, 3))
        expressions = [expression.xreplace(definitions) for expression in tangent]
        arguments = [s for row in reference for s in row] + list(adjugate) + [determinant, sp.Symbol("lmbda"), sp.Symbol("mu")]
        return sp.lambdify(arguments, expressions, "numpy")

    @staticmethod
    def _exact_tangent_function(flux_form):
        """`A_ijkl` as a function of the flux's own variable, by direct differentiation."""
        variables, flux = list(flux_form.gradient), list(flux_form.flux)
        tangent = [sp.diff(flux[i * 3 + j], variables[k * 3 + l]) for i, j, k, l in itertools.product(range(3), repeat=4)]
        return sp.lambdify(variables + [sp.Symbol("lmbda"), sp.Symbol("mu")], tangent, "numpy")

    def _actions(self, plan, point, exact_tangent, fold, shift):
        rng = np.random.default_rng(7)
        n, nq = plan.n_nodes, plan.n_setup_points
        lmbda, mu = 2.2, 7.0 / 3.0
        # A curved element: reference nodes mapped affinely, then perturbed.
        reference_nodes = self._reference_nodes(plan)
        J0 = np.array([[1.5, 0.2, 0.1], [0.1, 1.25, 0.3], [0.2, 0.1, 1.1]])
        nodes = reference_nodes @ J0.T + (0.0 if plan.element_type == "TET4" else 0.03 * rng.standard_normal((n, 3)))
        u = 0.05 * rng.standard_normal((3, n))
        h = rng.standard_normal((3, n))
        setup = plan.setup_gradients if plan.setup_gradients is not None else plan.gradients
        G = [np.array(setup[q]) for q in range(nq)]  # [m][j] at q

        # The stored tensors, from the point function.
        T = np.zeros((nq, 45))
        for q in range(nq):
            J = nodes.T @ G[q].T  # J[a][m] = dx_a/dxi_m
            adj = np.linalg.det(J) * np.linalg.inv(J)
            gu_ref = (u @ G[q].T).reshape(-1)  # [c][m]
            T[q] = plan.setup_weights[q] * np.array(point(*gu_ref, *adj.reshape(-1), np.linalg.det(J), lmbda, mu), dtype=float)
        if plan.is_moments:
            stored = np.array(plan.moment_factor).T @ T  # [t][c]
        else:
            stored = T.copy()
        layout = plan.fold_layout(True)
        if fold:
            unfolded = stored.copy()
            for t in range(plan.n_stored):
                for c in range(45):
                    d = layout.directions[c]
                    if d >= 0 and not ((layout.mask[t] >> d) & 1):
                        stored[t, c] += layout.coefficient[t][d] * unfolded[layout.source[t][d], c]
            for t in range(plan.n_stored):
                for c in range(45):
                    d = layout.directions[c]
                    if d >= 0 and (layout.mask[t] >> d) & 1:
                        stored[t, c] = 0.0

        def slot(r, c):
            return packed_slot(r, c, 9, True)

        def full(S):
            return np.array([[S[slot(r, c)] for c in range(9)] for r in range(9)])

        # The stored action: Y_b += S_t P_a (and swapped), out_i,p = sum_b sum_n DT[b][n][p] Y_b[i,n].
        P = [(h @ np.array(plan.gradients[r]).T).reshape(-1) for r in range(plan.n_gradients)]  # [k*3+m]
        Y = np.zeros((plan.n_gradients, 9))
        for t, (a, b) in enumerate(plan.pairs):
            S = full(stored[t])
            if fold:
                for r, c in itertools.product(range(9), repeat=2):
                    if r % 3 == c % 3 and (layout.mask[t] >> (r % 3)) & 1:
                        S[r, c] = 0.0
            Y[b] += S @ P[a]
            if a != b:
                Y[a] += S @ P[b]
        out = np.zeros((3, n))
        for r in range(plan.n_gradients):
            D = np.array(plan.gradients[r])  # [n][p]
            for i in range(3):
                out[i] += Y[r][i * 3:(i + 1) * 3] @ D

        # The exact action: sum_q w_q det(J) A_ijkl(grad u) d h_k/dx_l d phi_p/dx_j, no packing.
        exact = np.zeros((3, n))
        for q in range(nq):
            J = nodes.T @ G[q].T
            inverse = np.linalg.inv(J)
            physical = G[q].T @ inverse  # [j][a] = dphi_j/dx_a
            grad_u = u @ physical
            grad_h = h @ physical
            A = np.array(exact_tangent(*(shift + grad_u).reshape(-1), lmbda, mu), dtype=float).reshape(3, 3, 3, 3)
            exact += plan.setup_weights[q] * np.linalg.det(J) * np.einsum("ijkl,kl,pj->ip", A, grad_h, physical)
        return out, exact

    @staticmethod
    def _reference_nodes(plan):
        if plan.element_type == "TET4":
            return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        if plan.element_type == "TET10":
            v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
            return np.vstack([v] + [0.5 * (v[a] + v[b]) for a, b in edges])
        n1 = plan.line.n_shape
        x = np.linspace(0.0, 1.0, n1)
        lexicographic = np.array([[x[sx], x[sy], x[sz]] for sz in range(n1) for sy in range(n1) for sx in range(n1)])
        return lexicographic


if __name__ == "__main__":
    unittest.main()
