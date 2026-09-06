"""When an energy admits the compact metric kernel, and when it does not.

The compact P1 simplex kernel contracts through `FFF * grad`: six symmetric
components rather than nine adjugate ones and a determinant.  The residual path
selects it and the energy path does not, which is why declaring laplace as an
energy takes ten geometry streams where the residual takes six -- two thirds
more traffic on a memory-bound kernel.

`energy_gradient_metric_scale` is the question that has to be answered before
the energy path can make the same choice.  It is asked of the flux, because
that is the one object both formulations produce.
"""

import unittest

import sympy as sp

from codegen.framework.plans.form_transformations import (
    energy_gradient_metric_scale,
)
from codegen.framework.symbolic.weak_forms import sfem_soa_weak_form


class EnergyGradientMetricTest(unittest.TestCase):
    def test_a_scalar_potential_admits_the_metric(self):
        """`kappa/2 * ||grad u||^2` has flux `kappa * grad u`."""
        kappa = sp.Symbol("kappa")
        gradient = sp.Matrix(1, 3, sp.symbols("G[0:3]"))
        energy = kappa / 2 * sum(entry * entry for entry in gradient)
        self.assertEqual(
            energy_gradient_metric_scale(sfem_soa_weak_form(energy, gradient)),
            kappa,
        )

    def test_a_hyperelastic_energy_does_not(self):
        """Its first Piola depends on the gradient nonlinearly."""
        mu, lmbda = sp.symbols("mu lmbda")
        F = sp.Matrix(3, 3, sp.symbols("F[0:9]"))
        energy = mu / 2 * sum(f * f for f in F) + lmbda / 2 * (F.det() - 1) ** 2
        self.assertIsNone(
            energy_gradient_metric_scale(sfem_soa_weak_form(energy, F))
        )

    def test_a_square_variable_is_refused(self):
        """A deformation gradient is `I + grad(u)`, not the gradient.

        The weak form infers that from the shape, so a genuinely
        gradient-valued square variable is refused too.  That is the
        conservative direction -- it costs the specialization, not correctness
        -- and it stops being an inference when the qualifier reaches this
        layer.
        """
        kappa = sp.Symbol("kappa")
        F = sp.Matrix(3, 3, sp.symbols("F[0:9]"))
        energy = kappa / 2 * sum(f * f for f in F)
        self.assertIsNone(
            energy_gradient_metric_scale(sfem_soa_weak_form(energy, F))
        )

    def test_a_non_uniform_scale_is_refused(self):
        """Anisotropy does not factor through a single scalar."""
        gradient = sp.Matrix(1, 3, sp.symbols("G[0:3]"))
        energy = (
            gradient[0] ** 2 + 2 * gradient[1] ** 2 + 3 * gradient[2] ** 2
        ) / 2
        self.assertIsNone(
            energy_gradient_metric_scale(sfem_soa_weak_form(energy, gradient))
        )


if __name__ == "__main__":
    unittest.main()
