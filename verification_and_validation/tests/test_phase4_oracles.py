import sys
import unittest
from pathlib import Path

import numpy as np


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(SUITE_DIR / "kirsch_plate_hole_2d"))
from kirsch_plate_hole_2d.oracle import displacement as kirsch_displacement, polar_stress as kirsch_stress  # noqa: E402
from lame_spherical_shell_3d.oracle import displacement as lame_displacement, polar_stress as lame_stress  # noqa: E402
from inflated_spherical_shell_3d.oracle import (  # noqa: E402
    incompressible_shift,
    linear_moduli,
    nominal_stresses,
    radial_fields,
    small_strain_guess,
    solve_radial_shell,
)


class Phase4OracleTests(unittest.TestCase):
    def test_kirsch_stress_matches_displacement_derivatives(self):
        points = np.asarray(((1.25, 1.75), (2.25, 0.75), (1.5, 1.5)))
        h = 1.0e-6
        gradient = np.empty((len(points), 2, 2))
        for axis in range(2):
            offset = np.zeros(2)
            offset[axis] = h
            gradient[:, :, axis] = (
                kirsch_displacement(points + offset, 1.0, 1.0, 2.0, 3.0)
                - kirsch_displacement(points - offset, 1.0, 1.0, 2.0, 3.0)
            ) / (2.0 * h)
        strain = 0.5 * (gradient + np.swapaxes(gradient, -1, -2))
        sigma = 4.0 * strain + 3.0 * np.trace(strain, axis1=-2, axis2=-1)[:, None, None] * np.eye(2)
        direction = points / np.linalg.norm(points, axis=1)[:, None]
        tangent = np.stack((-direction[:, 1], direction[:, 0]), axis=-1)
        numerical = np.column_stack((
            np.einsum("ni,nij,nj->n", direction, sigma, direction),
            np.einsum("ni,nij,nj->n", tangent, sigma, tangent),
            np.einsum("ni,nij,nj->n", direction, sigma, tangent),
        ))
        np.testing.assert_allclose(numerical, kirsch_stress(points, 1.0, 1.0), atol=2.0e-9)
        hole_peak = kirsch_stress(np.asarray(((0.0, 1.0),)), 1.0, 1.0)[0]
        np.testing.assert_allclose(hole_peak, (0.0, 3.0, 0.0), atol=1.0e-14)

    def test_lame_pressure_and_free_outer_boundary(self):
        stress = lame_stress(np.asarray((1.0, 2.5)), 1.0, 2.5, 0.2)
        self.assertAlmostEqual(-0.2, stress[0, 0], places=14)
        self.assertAlmostEqual(0.0, stress[1, 0], places=14)
        points = np.asarray(((1.0, 0.0, 0.0), (0.0, 2.5, 0.0)))
        expected = lame_displacement(points, 1.0, 2.5, 0.2, 2.0, 3.0)
        self.assertGreater(expected[0, 0], 0.0)
        self.assertGreater(expected[1, 1], 0.0)
        self.assertEqual(0.0, expected[0, 1])

    def test_radial_bvp_small_strain_and_incompressible_limits(self):
        cases = (
            ("GeneratedNeoHookeanOgden", {"mu": 2.0, "lambda": 3.0},
             {"mu": 2.0, "lambda": 10000.0}),
            ("GeneratedModifiedMooneyRivlin", {"c1": 0.5, "c2": 0.5, "kappa": 5.0},
             {"c1": 0.5, "c2": 0.5, "kappa": 10000.0}),
        )
        for operator, material, near in cases:
            with self.subTest(operator=operator):
                mu, bulk = linear_moduli(operator, material)
                reference = np.asarray((1.0, 2.5))
                solution = solve_radial_shell(operator, 1.0, 2.5, 1.0e-4, material)
                numerical = radial_fields(solution, reference, operator, material)
                linear_radius = small_strain_guess(reference, 1.0, 2.5, 1.0e-4, mu, bulk)[0]
                relative = np.max(np.abs((numerical[:, 0] - linear_radius) / (linear_radius - reference)))
                self.assertLess(relative, 1.0e-4)
                self.assertAlmostEqual(-1.0e-4, numerical[0, 4], places=8)
                self.assertAlmostEqual(0.0, numerical[1, 4], places=8)

                pressure = 0.05
                near_solution = solve_radial_shell(operator, 1.0, 2.5, pressure, near)
                near_fields = radial_fields(near_solution, reference, operator, near)
                shift = incompressible_shift(operator, 1.0, 2.5, pressure, near)
                incompressible_radius = (reference ** 3 + shift) ** (1.0 / 3.0)
                relative = np.max(np.abs(
                    (near_fields[:, 0] - incompressible_radius) / (incompressible_radius - reference)
                ))
                self.assertLess(relative, 4.0e-4)
                self.assertTrue(np.all(near_fields[:, 3] > 0.0))

    def test_radial_nominal_stresses_are_zero_at_identity(self):
        for operator, material in (
            ("GeneratedNeoHookeanOgden", {"mu": 2.0, "lambda": 3.0}),
            ("GeneratedModifiedMooneyRivlin", {"c1": 0.5, "c2": 0.5, "kappa": 5.0}),
        ):
            radial, tangential = nominal_stresses(operator, 1.0, 1.0, material)
            self.assertAlmostEqual(0.0, radial, places=13)
            self.assertAlmostEqual(0.0, tangential, places=13)


if __name__ == "__main__":
    unittest.main()
