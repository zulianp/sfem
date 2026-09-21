"""An energy form's history stream is a stream, not a material constant.

An energy form classifies its free symbols into the current state, the
direction, and -- everything else -- material parameters.  That last bucket is
a catch-all, and it used to swallow the history: a `gen.dt(u)` written inside
an energy leaves `u_old` and `u_old_grad[i]` in the expression, and they were
handed to the wrapper as scalars a caller sets per block.  The failure that
follows is a runtime one, `require_real_value("u_old_grad[0]")` looking for a
number that is really a field.

The residual front end never had the problem because it builds the previous
symbols itself and asks for them by name.  The energy front end has no such
list, so the question is asked of the spelling, by the one function that
produces it -- the same arrangement `is_time_rate_shift` already uses.

No material in the tree carries a rate inside an energy today, so these tests
are the only thing holding the classification.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from codegen.framework.forms.equations import _expression_dependencies  # noqa: E402
from sfem import gen  # noqa: E402
from codegen.framework.symbolic.fields import is_previous_symbol  # noqa: E402


class PreviousSymbolSpellingTest(unittest.TestCase):
    def test_recognises_what_previous_function_produces(self):
        for name in ("u_old", "p_old", "u_old_grad[0]", "velocity_old_grad[2]"):
            self.assertTrue(is_previous_symbol(name), name)

    def test_leaves_every_other_kind_alone(self):
        # The shift is a genuine material constant -- it answers from the
        # parameters when no scheme is attached -- so it must not be swept into
        # the history stream with the vectors.
        for name in ("u", "p", "u_grad[1]", "mu", "lmbda", "u_dt_shift",
                     "u_direction_grad[2]", "u_test_grad[0]"):
            self.assertFalse(is_previous_symbol(name), name)


class EnergyFormDependencyTest(unittest.TestCase):
    def _rate_energy(self, dim=2):
        element = gen.VectorElement("Lagrange", degree=1)
        V = gen.FunctionSpace(element)
        with gen.geometric_dimension_context(dim):
            u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
            rate = gen.grad(gen.dt(u))
            return gen.inner(rate, rate) / 2, u

    def test_history_is_a_stream_and_not_a_parameter(self):
        expression, _ = self._rate_energy()
        dependencies = _expression_dependencies(expression)

        self.assertTrue(dependencies.previous)
        self.assertTrue(dependencies.previous_symbols)

        parameters = {str(p) for p in dependencies.parameters}
        for symbol in dependencies.previous_symbols:
            self.assertNotIn(str(symbol), parameters)
        self.assertFalse(
            any("_old" in name for name in parameters),
            "history symbols reached the material parameters: %s" % sorted(parameters),
        )

    def test_the_shift_stays_a_parameter(self):
        expression, _ = self._rate_energy()
        dependencies = _expression_dependencies(expression)
        parameters = {str(p) for p in dependencies.parameters}
        self.assertIn("u_dt_shift", parameters)

    def test_a_rate_free_energy_declares_no_history(self):
        element = gen.VectorElement("Lagrange", degree=1)
        V = gen.FunctionSpace(element)
        with gen.geometric_dimension_context(2):
            u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
            grad_u = gen.grad(u)
            dependencies = _expression_dependencies(gen.inner(grad_u, grad_u) / 2)
        self.assertFalse(dependencies.previous)
        self.assertEqual(dependencies.previous_symbols, ())


if __name__ == "__main__":
    unittest.main()
