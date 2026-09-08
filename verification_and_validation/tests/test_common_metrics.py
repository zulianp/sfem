import sys
import unittest
from pathlib import Path

import numpy as np


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))

from common.convergence import fit_spatial_convergence, fit_temporal_convergence  # noqa: E402
from common.metrics import (  # noqa: E402
    curve_errors,
    max_abs_error,
    relative_l2_error,
    weighted_l2_error,
    weighted_relative_l2_error,
)


class MetricTests(unittest.TestCase):
    def test_relative_norm_uses_absolute_floor(self):
        observed = np.asarray((3.0e-9, 4.0e-9))
        expected = np.zeros(2)
        self.assertAlmostEqual(0.5, relative_l2_error(observed, expected, absolute_floor=1.0e-8))

    def test_weighted_norm_and_floor(self):
        observed = np.asarray((1.0, 3.0))
        expected = np.asarray((1.0, 1.0))
        weights = np.asarray((1.0, 0.25))
        self.assertAlmostEqual(1.0, weighted_l2_error(observed, expected, weights))
        self.assertAlmostEqual(
            1.0 / np.sqrt(1.25), weighted_relative_l2_error(observed, expected, weights)
        )

    def test_curve_error_interpolates_reference(self):
        reference_x = np.asarray((0.0, 0.5, 1.0))
        reference_y = 2.0 * reference_x + 1.0
        observed_x = np.asarray((0.0, 0.25, 0.75, 1.0))
        observed_y = 2.0 * observed_x + 1.0
        errors = curve_errors(observed_x, observed_y, reference_x, reference_y)
        self.assertEqual(4, errors["sample_count"])
        self.assertEqual(0.0, errors["absolute_l2"])
        self.assertEqual(0.0, errors["relative_l2"])
        self.assertEqual(0.0, max_abs_error(observed_y, 2.0 * observed_x + 1.0))


class ConvergenceTests(unittest.TestCase):
    def test_spatial_quadratic_rate(self):
        sizes = np.asarray((0.5, 0.25, 0.125, 0.0625))
        fit = fit_spatial_convergence(sizes, 3.0 * sizes ** 2)
        self.assertAlmostEqual(2.0, fit.rate)
        self.assertAlmostEqual(1.0, fit.r_squared)
        self.assertEqual(4, fit.sample_count)

    def test_temporal_first_order_rate(self):
        time_steps = np.asarray((0.2, 0.1, 0.05))
        fit = fit_temporal_convergence(time_steps, 0.75 * time_steps)
        self.assertAlmostEqual(1.0, fit.rate)
        self.assertAlmostEqual(1.0, fit.r_squared)

    def test_zero_error_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "errors must be finite and positive"):
            fit_spatial_convergence((0.5, 0.25), (0.1, 0.0))


if __name__ == "__main__":
    unittest.main()
