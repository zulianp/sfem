"""Prony relaxation as an extension of any hyperelastic energy.

`PronyRelaxation.relax(psi_r, F)` turns a relaxed energy into its step-energy
contribution, gamma * psi_r + 1/2 S_hist : (C - I), and publishes S_r, the
stress the history update reads, as an integration-point output.  These tests
hold the three facts the rest of the pipeline relies on:

  - the step energy's first Piola-Kirchhoff stress is F (gamma S_r + S_hist),
    so the generated gradient, apply and assembly are the algorithmic ones;
  - the published output is S_r = 2 dpsi_r/dC, symmetric, in the stored order;
  - the history's components are an integration-point stream to the form
    layer, never material parameters, and the step weight is a parameter the
    operator answers from its relaxation object.
"""

import os
import random
import sys
import unittest

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from codegen.framework.forms.equations import _expression_dependencies  # noqa: E402
from codegen.framework.plans import quadrature_fields  # noqa: E402
from codegen.framework.plans.form_emission import (  # noqa: E402
    FormAccumulation,
    form_accumulation,
    objective_kernel_variants,
    writes_per_shape,
)
from codegen.framework.symbolic.fields import (  # noqa: E402
    is_quadrature_symbol,
    quadrature_symbol_parts,
    symmetric_tensor_entries,
)
from codegen.framework.symbolic.relaxation import (  # noqa: E402
    is_relaxation_gamma,
    relaxation_group_of_gamma,
)
from sfem import gen  # noqa: E402


C10 = sp.Symbol("C10")
C01 = sp.Symbol("C01")


def _variable(dim):
    return gen.variable(sp.Matrix(dim, dim, lambda i, j: sp.Symbol("u%d%d" % (i, j))), name="F")


def _mooney_rivlin_isochoric(F, dim):
    I1, I2, J = gen.isochoric_invariants(F, dim)
    return C10 * (J ** sp.Rational(-2, 3) * I1 - 3) + C01 * (J ** sp.Rational(-4, 3) * I2 - 3)


def _random_deformation(dim, seed):
    rng = random.Random(seed)
    return sp.Matrix(dim, dim, lambda i, j: (1 if i == j else 0) + sp.Rational(rng.randint(-200, 200), 1000))


class RelaxTest(unittest.TestCase):
    def _relaxed(self, dim):
        F = _variable(dim)
        group = gen.PronyRelaxation("iso")
        psi_r = _mooney_rivlin_isochoric(F, dim)
        return F, group, psi_r, group.relax(psi_r, F)

    def _at(self, F, value, expression):
        return sp.sympify(expression).xreplace(dict(zip(tuple(F.value), tuple(value))))

    def test_the_step_energy_differentiates_to_the_algorithmic_stress(self):
        for dim in (2, 3):
            F, group, psi_r, W = self._relaxed(dim)
            history = sp.Matrix(dim, dim, lambda i, j: 0)
            symbols = {str(s): s for s in sp.sympify(W).free_symbols}
            for k, (i, j) in enumerate(symmetric_tensor_entries(dim)):
                history[i, j] = history[j, i] = symbols["iso_history_qp[%d]" % k]

            P_step = sp.Matrix(dim, dim, lambda i, j: sp.diff(W, F.value[i, j]))
            P_relaxed = sp.Matrix(dim, dim, lambda i, j: sp.diff(psi_r, F.value[i, j]))
            expected = group.gamma * P_relaxed + F.value * history

            values = {C10: sp.Rational(3, 10), C01: sp.Rational(1, 20), group.gamma: sp.Rational(9, 10)}
            values.update({s: sp.Rational(k + 1, 7) for k, s in enumerate(history) if s != 0})
            point = _random_deformation(dim, dim)
            difference = (P_step - expected).xreplace(values)
            for entry in difference:
                self.assertLess(abs(float(self._at(F, point, entry))), 1e-12)

    def test_the_output_is_the_relaxed_second_piola_kirchhoff_stress(self):
        for dim in (2, 3):
            F, group, psi_r, _W = self._relaxed(dim)
            ((name, components),) = group.quadrature_outputs().items()
            self.assertEqual(name, "iso_stress")
            self.assertEqual(len(components), len(symmetric_tensor_entries(dim)))

            # S = 2 dpsi/dC, by the chain rule through C = F^T F: dpsi/dF = F S.
            point = _random_deformation(dim, 10 + dim)
            values = {C10: sp.Rational(3, 10), C01: sp.Rational(1, 20)}
            P = sp.Matrix(dim, dim, lambda i, j: self._at(F, point, sp.diff(psi_r, F.value[i, j]).xreplace(values)))
            S = sp.zeros(dim, dim)
            for (i, j), component in zip(symmetric_tensor_entries(dim), components):
                S[i, j] = S[j, i] = self._at(F, point, component.xreplace(values))
            residual = point * S - P
            for entry in residual:
                self.assertLess(abs(float(entry)), 1e-12)

    def test_a_group_relaxes_one_energy(self):
        F, group, psi_r, _W = self._relaxed(3)
        with self.assertRaises(ValueError):
            group.relax(psi_r, F)


class ClassificationTest(unittest.TestCase):
    def test_history_components_are_a_stream_and_not_parameters(self):
        F = _variable(3)
        group = gen.PronyRelaxation("iso")
        W = group.relax(_mooney_rivlin_isochoric(F, 3), F)
        dependencies = _expression_dependencies(W, variables=(F,))
        parameters = {str(p) for p in dependencies.parameters}
        self.assertEqual(parameters, {"C10", "C01", "iso_relaxation_gamma"})
        self.assertEqual(len(dependencies.quadrature_symbols), 6)
        self.assertTrue(all(is_quadrature_symbol(s) for s in dependencies.quadrature_symbols))

    def test_the_spellings_have_one_owner_each(self):
        self.assertEqual(quadrature_symbol_parts("iso_history_qp[4]"), ("iso_history", 4))
        self.assertFalse(is_quadrature_symbol("u_old_grad[0]"))
        self.assertTrue(is_relaxation_gamma("iso_relaxation_gamma"))
        self.assertFalse(is_relaxation_gamma("u_dt_shift"))
        self.assertEqual(relaxation_group_of_gamma("vol_relaxation_gamma"), "vol")

    def test_the_stored_order_is_diagonal_then_upper_triangle(self):
        self.assertEqual(symmetric_tensor_entries(2), ((0, 0), (1, 1), (0, 1)))
        self.assertEqual(
            symmetric_tensor_entries(3), ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        )


class _Form:
    def __init__(self, name, dependencies=None, weak_form=None):
        self.name = name
        self.dependencies = dependencies
        self.weak_form = weak_form


class StreamPlanTest(unittest.TestCase):
    def test_a_quadrature_output_writes_per_point(self):
        form = _Form("quadrature_output")
        self.assertIs(form_accumulation(form), FormAccumulation.PER_POINT)
        self.assertFalse(writes_per_shape(form))
        self.assertEqual(objective_kernel_variants(form, True), ("plain",))

    def test_reads_then_writes_and_one_stride(self):
        F = _variable(3)
        group = gen.PronyRelaxation("iso")
        W = group.relax(_mooney_rivlin_isochoric(F, 3), F)
        dependencies = _expression_dependencies(W, variables=(F,))

        class _WeakForm:
            quadrature_outputs = tuple(group.quadrature_outputs().items())

        form = _Form("quadrature_output", dependencies, _WeakForm())
        streams = quadrature_fields.quadrature_field_streams(form)
        self.assertEqual([(s.name, s.n_components, s.writes) for s in streams],
                         [("iso_history", 6, False), ("iso_stress", 6, True)])
        self.assertEqual(
            quadrature_fields.block_parameters(streams),
            (
                "const s_t *const RSTR iso_history",
                "s_t *const RSTR iso_stress",
                "const ptrdiff_t qp_stride",
            ),
        )
        self.assertEqual(quadrature_fields.block_parameters(()), ())
        (load,) = quadrature_fields.load_lines(streams, "", "lane")
        self.assertTrue(load.startswith("const s_t iso_history_qp[6] = {iso_history_q[lane], "))
        self.assertEqual(
            quadrature_fields.store_target(streams[1], 2, quadrature_fields.SINGLE_ELEMENT),
            "iso_stress_q[2 * NQ * qp_stride]",
        )


if __name__ == "__main__":
    unittest.main()
