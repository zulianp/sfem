"""An energy may be differentiated against a rate as well as a state.

A hyperelastic energy is differentiated against one staged quantity, the
deformation gradient, and the flux that falls out contracts straight against the
test gradient because `dF = grad(du)`.  An incremental variational formulation
-- `psi(u) + dt * phi(u_dot)`, which is what a Rayleigh dissipation or a
Kelvin-Voigt viscosity is as a potential -- needs a second group, the rate
`Fdot = shift * grad(u) + grad(u_old)`.  It depends on the same unknown, but by
`shift` rather than by 1, so the flux becomes a weighted sum and the weights are
not equal.

The weights are read from each group's recorded definition rather than declared,
so a material writes `gen.variable(gen.grad(gen.dt(u)), name="Fdot")` and nothing
else.  What reaches the plans is shaped exactly as a single group's flux would
be, which is what keeps the layers below unaware that any of this happened.

No material in the tree has a rate inside an energy yet, so these tests are the
only thing holding the arithmetic.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import sympy as sp  # noqa: E402

from codegen.framework.forms.equations import (  # noqa: E402
    _default_direction_symbols,
    _symbols_from_variables,
)
from codegen.framework.forms.forms import (  # noqa: E402
    FormOrder,
    energy_form_pipeline,
    energy_variable_factors,
)
from sfem import gen  # noqa: E402


def _rate_energy(dim=2):
    with gen.geometric_dimension_context(dim):
        V = gen.FunctionSpace(gen.VectorElement("Lagrange", degree=1))
        u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
        F = gen.variable(gen.Identity(dim) + gen.grad(u), name="F")
        Fdot = gen.variable(gen.grad(gen.dt(u)), name="Fdot")
        a, b = sp.symbols("a b")
        energy = a * gen.inner(F, F) / 2 + b * gen.inner(Fdot, Fdot) / 2
    return energy, F, Fdot, a, b


class EnergyVariableFactorTest(unittest.TestCase):
    def test_a_state_weighs_one_and_a_rate_weighs_the_shift(self):
        _, F, Fdot, _, _ = _rate_energy()
        self.assertEqual(
            energy_variable_factors((F, Fdot)),
            (sp.Integer(1), sp.Symbol("u_dt_shift")),
        )

    def test_a_single_group_takes_the_unchanged_path(self):
        _, F, _, _, _ = _rate_energy()
        self.assertIsNone(energy_variable_factors((F,)))

    def test_a_group_with_no_definition_is_refused(self):
        # Rather than silently weighing it 1, which would drop the shift and
        # leave a form that looks right and integrates the wrong problem.
        _, F, _, _, _ = _rate_energy()
        with self.assertRaises(ValueError):
            energy_variable_factors((F, sp.Symbol("bare")))


class EnergyRateFluxTest(unittest.TestCase):
    def _forms(self, dim=2):
        energy, F, Fdot, a, b = _rate_energy(dim)
        flat = _symbols_from_variables((F, Fdot))
        directions = _default_direction_symbols(flat)
        pipeline = energy_form_pipeline(energy, flat, directions, variable_groups=(F, Fdot))
        return pipeline, directions, a, b, dim * dim

    def test_the_flux_is_the_weighted_sum(self):
        pipeline, _, a, b, width = self._forms()
        flux = pipeline.form(FormOrder.ONE).expression
        shift = sp.Symbol("u_dt_shift")
        state = sp.symbols("F[0:%d]" % width)
        rate = sp.symbols("Fdot[0:%d]" % width)
        self.assertEqual(flux.shape[0], width)
        for i in range(width):
            self.assertEqual(sp.simplify(flux[i] - (a * state[i] + shift * b * rate[i])), 0)

    def test_the_flux_keeps_a_single_group_s_shape(self):
        # The point of folding here: two groups of `width` derivatives leave as
        # `width` flux entries, so nothing downstream sees a wider form.
        pipeline, _, _, _, width = self._forms()
        self.assertEqual(pipeline.form(FormOrder.ONE).expression.shape[0], width)

    def test_the_action_carries_the_weight_at_both_ends(self):
        # d/de of the flux along the field direction is
        # (a * 1 * 1 + b * shift * shift) * dF: the flux weighs each group, and
        # each group moves with the field by the same weight.
        pipeline, directions, a, b, width = self._forms()
        action = pipeline.form(FormOrder.TWO).expression
        shift = sp.Symbol("u_dt_shift")
        self.assertEqual(action.shape[0], width)
        for i in range(width):
            expected = (a + b * shift ** 2) * directions[i]
            self.assertEqual(sp.simplify(action[i] - expected), 0)

    def test_it_holds_in_three_dimensions_too(self):
        pipeline, _, a, b, width = self._forms(dim=3)
        flux = pipeline.form(FormOrder.ONE).expression
        shift = sp.Symbol("u_dt_shift")
        state = sp.symbols("F[0:%d]" % width)
        rate = sp.symbols("Fdot[0:%d]" % width)
        self.assertEqual(flux.shape[0], width)
        for i in range(width):
            self.assertEqual(sp.simplify(flux[i] - (a * state[i] + shift * b * rate[i])), 0)


if __name__ == "__main__":
    unittest.main()
