"""The two front ends already name their kernels the same way.

`add_energy` declares kernels ("objective", "gradient", "apply") and
`add_residual` declares ("gradient", "apply").  Those are one sequence under
two vocabularies:

    objective   0-form   the energy, or a residual-based merit
    gradient    1-form   the gradient, or the negated residual
    apply       2-form   the Hessian action, or the Jacobian action

This pins that correspondence, because it is the thing the energy and residual
paths have to agree on before anything below the form layer can stop
distinguishing them.  Emission used to ask `form.name == "objective"` twenty
times; a string comparison in an emitter is not an agreement, it is two
emitters happening to spell the same word.
"""

import unittest

from codegen.framework.plans.form_emission import (
    FORM_ORDER_BY_KERNEL,
    form_order,
    writes_per_shape,
)
from codegen.framework.symbolic.forms import FormOrder


class Form:
    def __init__(self, name):
        self.name = name


class FormEmissionVocabularyTest(unittest.TestCase):
    maxDiff = None

    def test_the_three_kernels_are_the_three_orders(self):
        self.assertEqual(
            {name: order for name, order in FORM_ORDER_BY_KERNEL.items()},
            {
                "objective": FormOrder.ZERO,
                "gradient": FormOrder.ONE,
                "apply": FormOrder.TWO,
            },
        )

    def test_both_front_ends_declare_names_this_covers(self):
        """Whatever a material can ask for, this must be able to order."""
        import inspect

        from codegen.framework.symbolic import equations

        source = inspect.getsource(equations)
        declared = set()
        for line in source.splitlines():
            if "kernels=(" not in line:
                continue
            body = line.split("kernels=(", 1)[1]
            declared.update(
                part.strip().strip('",)') for part in body.split(",") if '"' in part
            )
        declared.discard("")
        self.assertTrue(declared, "no kernel names found to check against")
        for name in declared:
            with self.subTest(kernel=name):
                self.assertIn(name, FORM_ORDER_BY_KERNEL)

    def test_only_the_zero_form_accumulates_a_scalar(self):
        self.assertFalse(writes_per_shape(Form("objective")))
        self.assertTrue(writes_per_shape(Form("gradient")))
        self.assertTrue(writes_per_shape(Form("apply")))

    def test_an_undeclared_kernel_is_refused_rather_than_guessed(self):
        """A new kernel name is a change to the form algebra, not a string."""
        with self.assertRaises(ValueError):
            form_order(Form("hessian_crs"))

    def test_emission_reads_this_rather_than_comparing_strings(self):
        from codegen.framework.emitters import energy_codegen

        self.assertIs(energy_codegen.writes_per_shape, writes_per_shape)
        self.assertNotIn(
            'form.name == "objective"',
            open(energy_codegen.__file__, encoding="utf-8").read(),
            "emission is back to comparing the kernel name to a string",
        )


if __name__ == "__main__":
    unittest.main()


class FormContractionTest(unittest.TestCase):
    """The contraction a form uses, and the table that selects on it."""

    maxDiff = None

    def test_a_weak_form_defers_its_flux(self):
        from codegen.framework.plans.form_emission import (
            FormContraction,
            form_contraction,
        )

        class WithWeakForm:
            weak_form = object()

        class WithoutWeakForm:
            weak_form = None

        self.assertIs(
            form_contraction(WithWeakForm()), FormContraction.DEFERRED_FLUX
        )
        self.assertIs(
            form_contraction(WithoutWeakForm()), FormContraction.POINTWISE
        )

    def test_both_block_emitters_take_the_same_arguments(self):
        """Interchangeability is the point; a table cannot select otherwise.

        This is what "no distinction below the form layer" means concretely
        here: the two paths differ in what they emit, not in how they are
        called, so which one runs is a lookup rather than a branch.
        """
        import inspect

        from codegen.framework.emitters import energy_codegen

        signatures = {
            name: inspect.signature(getattr(energy_codegen, name))
            for name in (
                "_sfem_soa_weak_form_block_function",
                "_sfem_soa_pointwise_block_function",
            )
        }
        parameters = {
            name: list(signature.parameters) for name, signature in signatures.items()
        }
        self.assertEqual(
            parameters["_sfem_soa_weak_form_block_function"],
            parameters["_sfem_soa_pointwise_block_function"],
        )

    def test_the_table_covers_every_contraction(self):
        from codegen.framework.emitters import energy_codegen
        from codegen.framework.plans.form_emission import FormContraction

        self.assertEqual(
            set(energy_codegen._BLOCK_FUNCTION_BY_CONTRACTION),
            set(FormContraction),
            "a contraction with no entry would fail at generation, not here",
        )
