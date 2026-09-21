"""Substituting a trial basis function for the direction gives the same column.

That equality is the whole licence for not probing.  If it holds, an element
matrix can be built in one pass with the state work done once per quadrature
point, instead of one apply per trial degree of freedom -- thirty per element on
TET10.

Measured on the Mooney-Rivlin Kelvin-Voigt Newmark viscous flux, which is the
form that still probes: linear in the direction, and the largest disagreement
between probing and substituting over every flux component and every trial
component was 3.55e-15.  That run takes minutes of sympy, so what is pinned here
is the machinery, on forms small enough to check exactly.
"""

import unittest

import sympy as sp

from codegen.framework.plans.direct_assembly import (
    direction_gradient_symbols,
    flux_is_linear_in_the_direction,
    trial_direction_substitution,
)


FIELDS = ("u0", "u1", "u2")
DIM = 3


def _linear_flux(coefficients):
    """A flux linear in the direction, with known coefficients."""
    symbols = direction_gradient_symbols(FIELDS, DIM)
    state = sp.Symbol("state")
    return sum(
        coefficients[c][e] * state * symbols[c][e]
        for c in range(DIM)
        for e in range(DIM)
    )


class TrialSubstitutionTest(unittest.TestCase):
    def test_substituting_equals_probing(self):
        """The claim, stated as an equality and checked exactly.

        Probing sets every direction component to zero but one and reads the
        answer; substituting puts the trial gradient in that component at
        generation time.  Same column, and here with no round-off to hide in.
        """
        coefficients = [[sp.Rational(c + 2, e + 3) for e in range(DIM)] for c in range(DIM)]
        flux = _linear_flux(coefficients)
        symbols = direction_gradient_symbols(FIELDS, DIM)
        gradient = [sp.Symbol("trial_grad%d" % e) for e in range(DIM)]

        for trial_component in range(DIM):
            probed = flux.subs(
                {s: sp.Integer(0) for row in symbols for s in row}
            ).subs(
                {symbols[trial_component][e]: gradient[e] for e in range(DIM)}
            )
            # `probe` zeroes everything first, so rebuild it the way a kernel would
            probe = {s: sp.Integer(0) for row in symbols for s in row}
            probe.update({symbols[trial_component][e]: gradient[e] for e in range(DIM)})
            probed = flux.subs(probe)
            direct = flux.subs(
                trial_direction_substitution(FIELDS, DIM, trial_component, gradient)
            )
            with self.subTest(trial_component=trial_component):
                self.assertEqual(sp.simplify(probed - direct), 0)

    def test_the_other_components_go_to_zero(self):
        """A basis function for one field has no gradient in the others."""
        gradient = [sp.Symbol("trial_grad%d" % e) for e in range(DIM)]
        substitution = trial_direction_substitution(FIELDS, DIM, 1, gradient)
        symbols = direction_gradient_symbols(FIELDS, DIM)
        for component in range(DIM):
            for axis in range(DIM):
                expected = gradient[axis] if component == 1 else sp.Integer(0)
                self.assertEqual(substitution[symbols[component][axis]], expected)

    def test_a_nonlinear_flux_is_refused(self):
        """The failure mode with no other symptom.

        A flux that is not linear in the direction substitutes perfectly happily
        and yields a wrong matrix -- nothing downstream would notice, because
        the result is still a plausible number in the right place.  So the
        linearity is asserted rather than assumed.
        """
        symbols = direction_gradient_symbols(FIELDS, DIM)
        linear = _linear_flux([[sp.Integer(1)] * DIM for _ in range(DIM)])
        self.assertTrue(flux_is_linear_in_the_direction(linear, FIELDS, DIM))

        squared = linear + symbols[0][0] * symbols[1][2]
        self.assertFalse(flux_is_linear_in_the_direction(squared, FIELDS, DIM))

        self.assertTrue(
            flux_is_linear_in_the_direction(sp.Symbol("state_only"), FIELDS, DIM),
            "a flux that does not mention the direction at all is linear in it",
        )


if __name__ == "__main__":
    unittest.main()
