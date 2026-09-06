"""Both front ends reach the same flux.

The prescribed architecture says an energy formulation and a residual
formulation lower to the same forms, and that nothing below the form layer can
tell them apart.  The obstacle was concrete: sum factorization contracts a
*flux* against test gradients, an energy has one by differentiation, and a
weak form has already contracted so it appeared not to.

It does.  A weak form is linear in its test function by construction, so
differentiating it against the test symbols returns the coefficient exactly.
These tests hold that: the extraction is exact for every residual material in
the tree, and for the one operator written both ways -- `laplace` as a
residual, `scalar_potential` as the energy whose gradient it is -- the two
front ends produce the same flux, symbol for symbol.
"""

import importlib
import os
import sys
import unittest

import sympy as sp

from codegen.framework.symbolic.weak_forms import (
    SfemSoAFluxForm,
    flux_form_from_energy,
    flux_form_from_residual,
    sfem_soa_weak_form,
)


MATERIALS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials"
)

#: Every material written as a residual.  `neumann_general` is absent because
#: it does not expose a `systems` attribute to import from.
RESIDUAL_MATERIALS = (
    "laplace",
    "stokes",
    "navier_stokes",
    "two_phase_flow",
    "neumann",
)


def _collection(name, index):
    if MATERIALS not in sys.path:
        sys.path.insert(0, MATERIALS)
    module = importlib.import_module(name)
    system = list(module.systems.systems)[index]
    return system, system.form_collection(list(system.equations)[0])


class FluxFormTest(unittest.TestCase):
    def test_every_residual_material_gives_its_flux_back_exactly(self):
        for name in RESIDUAL_MATERIALS:
            for index in (0, -1):
                system, collection = _collection(name, index)
                for expression, field in zip(
                    collection.residual_expressions, collection.residual_fields
                ):
                    form = flux_form_from_residual(
                        (expression,), (field,), system.dim
                    )
                    rebuilt = form.weak_form_expression(
                        field.test_gradient, field.test_value
                    )
                    with self.subTest(material=name, dim=system.dim, field=field.name):
                        self.assertEqual(
                            sp.simplify(sp.expand(sp.sympify(expression)) - rebuilt),
                            0,
                        )

    def test_the_same_operator_written_both_ways_has_one_flux(self):
        system, residual = _collection("laplace", -1)
        from_residual = flux_form_from_residual(
            residual.residual_expressions, residual.residual_fields, system.dim
        )

        energy_system, energy = _collection("scalar_potential", -1)
        weak_form = sfem_soa_weak_form(
            energy.forms[0].expression,
            sp.Matrix(1, energy_system.dim, list(energy.variables)),
        )
        from_energy = flux_form_from_energy(weak_form)

        # The two name their gradient symbols differently; the flux is the same
        # function of them, which is what "the same operator" means here.
        rename = dict(zip(from_energy.gradient, from_residual.gradient))
        for energy_entry, residual_entry in zip(from_energy.flux, from_residual.flux):
            self.assertEqual(
                sp.simplify(energy_entry.subs(rename) - residual_entry), 0
            )
        self.assertFalse(from_energy.has_source)
        self.assertFalse(from_residual.has_source)

    def test_a_pure_diffusion_has_no_source_and_a_mixed_form_does(self):
        system, laplace = _collection("laplace", -1)
        self.assertFalse(
            flux_form_from_residual(
                laplace.residual_expressions, laplace.residual_fields, system.dim
            ).has_source
        )
        system, stokes = _collection("stokes", -1)
        self.assertTrue(
            flux_form_from_residual(
                stokes.residual_expressions, stokes.residual_fields, system.dim
            ).has_source
        )

    def test_a_form_that_is_not_linear_in_its_test_function_is_refused(self):
        test_grad = sp.symbols("u_test_grad_0 u_test_grad_1")
        gradient = sp.symbols("u_grad_0 u_grad_1")

        class Field(object):
            name = "u"

        field = Field()
        field.test_gradient = test_grad
        field.test_value = None
        field.gradient = gradient

        with self.assertRaises(ValueError) as raised:
            flux_form_from_residual((test_grad[0] ** 2,), (field,), 2)
        self.assertIn("not linear", str(raised.exception))

    def test_the_shape_is_checked_rather_than_assumed(self):
        with self.assertRaises(ValueError):
            SfemSoAFluxForm((sp.Integer(1),), (sp.Integer(0),), (sp.Symbol("g"),), 2, 1)


if __name__ == "__main__":
    unittest.main()
