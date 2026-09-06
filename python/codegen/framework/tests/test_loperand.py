"""What the loperand isolates, for every material in the tree.

The cached metric used to be gated on "the flux is a uniform scalar multiple of
the field gradient".  That is true of the Laplacian and of nothing else, so the
condition reached exactly one operator by construction and could never have
found a second.

`plans.loperand` asks the structural question instead: the affine element's
operator is a linear map from the reference-gradient coefficients to the
outputs, and the matrix of that map either is or is not free of the gradient
and symmetric.  These tests record what it finds, because the finding is the
point -- the numbers below are what say whether caching is worth it, and they
are the input to a cost decision rather than a yes or no.
"""

import importlib
import os
import sys
import unittest

import sympy as sp

from codegen.framework.plans.loperand import (
    gradient_metric_matrix,
    gradient_metric_scale,
    loperand_matrix,
)
from codegen.framework.symbolic.weak_forms import (
    flux_form_from_energy,
    flux_form_from_residual,
    sfem_soa_weak_form,
)


MATERIALS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials"
)


def _flux_form(name, dim_index=-1):
    if MATERIALS not in sys.path:
        sys.path.insert(0, MATERIALS)
    module = importlib.import_module(name)
    system = list(module.systems.systems)[dim_index]
    collection = system.form_collection(list(system.equations)[0])
    if collection.kind.value == "energy":
        weak_form = sfem_soa_weak_form(
            collection.forms[0].expression,
            sp.Matrix(
                len(collection.variables) // system.dim,
                system.dim,
                list(collection.variables),
            ),
        )
        return flux_form_from_energy(weak_form)
    return flux_form_from_residual(
        collection.residual_expressions, collection.residual_fields, system.dim
    )


class LoperandTest(unittest.TestCase):
    def test_a_scalar_diffusion_isolates_the_gradient_metric(self):
        for name in ("laplace", "scalar_potential"):
            isolated = loperand_matrix(_flux_form(name))
            with self.subTest(material=name):
                self.assertTrue(isolated.symmetric)
                self.assertEqual(isolated.order, 3)
                self.assertEqual(isolated.components, 6)
                self.assertEqual(isolated.carries, ("kappa",))
                # Not merely the right size: the matrix is kappa * FFF exactly.
                metric = gradient_metric_matrix(3)
                for row in range(3):
                    for column in range(3):
                        self.assertEqual(
                            sp.simplify(
                                isolated.entry(row, column)
                                - sp.Symbol("kappa") * metric[row, column]
                            ),
                            0,
                        )
                self.assertEqual(gradient_metric_scale(isolated), sp.Symbol("kappa"))

    def test_the_parameters_ride_in_the_isolated_matrix(self):
        """A cache holds the operator, not the mesh.

        `kappa` is in the isolated matrix rather than beside it, which is what
        makes an element-varying coefficient representable: it rides in the
        same object the geometry does instead of disqualifying it.
        """
        self.assertEqual(loperand_matrix(_flux_form("laplace")).carries, ("kappa",))
        self.assertEqual(
            loperand_matrix(_flux_form("linear_elasticity")).carries,
            ("lmbda", "mu"),
        )

    def test_a_vector_field_isolates_too_but_is_not_worth_caching(self):
        """Capability and cost are different questions, and this separates them.

        Linear elasticity's map is isolatable and symmetric -- there is nothing
        about it that forbids a cache.  Forty-five numbers per element against
        the adjugate's nine and a determinant is why it does not get one, and
        that is a comparison the caller can make only because the matcher
        reports the count rather than answering yes or no.
        """
        isolated = loperand_matrix(_flux_form("linear_elasticity"))
        self.assertTrue(isolated.symmetric)
        self.assertEqual(isolated.order, 9)
        self.assertEqual(isolated.components, 45)
        self.assertGreater(isolated.components, 3 * 3 + 1)
        # It is not the shape the six-stream ABI carries, so it takes the adjugate.
        self.assertIsNone(gradient_metric_scale(isolated))

    def test_a_nonlinear_flux_isolates_nothing(self):
        for name in ("neohookean_ogden", "saint_venant_kirchhoff"):
            with self.subTest(material=name):
                self.assertIsNone(loperand_matrix(_flux_form(name)))

    def test_a_state_dependent_map_is_isolatable_but_not_symmetric(self):
        """Two-phase flow, and why symmetry is the test rather than linearity.

        Its map does not depend on the *gradient*, so it isolates -- but it
        depends on the solution, and it is not symmetric.  Either would rule out
        the cached metric; the symmetry test catches it without the matcher
        having to know what a solution is.
        """
        isolated = loperand_matrix(_flux_form("two_phase_flow"))
        self.assertIsNotNone(isolated)
        self.assertFalse(isolated.symmetric)
        self.assertIn("p_w", isolated.carries)
        self.assertIsNone(gradient_metric_scale(isolated))


if __name__ == "__main__":
    unittest.main()
