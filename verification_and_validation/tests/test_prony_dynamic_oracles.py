import importlib.util
import unittest
from pathlib import Path
import sys

import numpy as np


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))

from common.prony import step_weights  # noqa: E402


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, SUITE_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARMONIC = load_module("prony_harmonic_oracle", "prony_harmonic_3d/oracle.py")
BAR = load_module("prony_bar_oracle", "prony_bar_mode_3d/oracle.py")
G, TAU = "0.3,0.25,0.2", "0.01,0.1,1.0"


class StepWeightTests(unittest.TestCase):
    def test_weights_agree_to_second_order_in_dt_over_tau(self):
        # linear: g (1 - x/2 + x^2/6 - ...), midpoint: g (1 - x/2 + x^2/8 - ...), x = dt/tau.
        g, tau = np.asarray([0.3, 0.2]), np.asarray([0.05, 0.3])
        for x in (1.0e-2, 1.0e-3):
            dt = x * tau[0]
            alpha, linear = step_weights(g, tau, dt, "linear")
            _, midpoint = step_weights(g, tau, dt, "midpoint")
            np.testing.assert_allclose(alpha, np.exp(-dt / tau), rtol=1.0e-15)
            np.testing.assert_allclose((linear - midpoint) / g, (dt / tau) ** 2 / 24.0, rtol=2.0 * x)


class HarmonicOracleTests(unittest.TestCase):
    def test_recursion_converges_to_the_continuum_response(self):
        omega, eps0 = 10.0, 1.0e-7
        errors = []
        for steps in (64, 128, 256):
            dt = 2.0 * np.pi / omega / steps
            t = dt * np.arange(4 * steps + 1)
            replayed = HARMONIC.algorithmic_reaction(eps0 * np.sin(omega * t), G, TAU, dt, 1.0, 0.5)
            exact = HARMONIC.continuum_linear_reaction(t, eps0, omega, G, TAU, 1.0, 0.5)
            errors.append(np.linalg.norm(replayed[1:] - exact[1:]) / np.linalg.norm(exact[1:]))
        self.assertGreater(np.log2(errors[0] / errors[1]), 1.9)
        self.assertGreater(np.log2(errors[1] / errors[2]), 1.9)

    def test_fit_recovers_the_transfer_function(self):
        for integrator in ("linear", "midpoint"):
            for omega in (0.1, 10.0, 1000.0):
                dt = 2.0 * np.pi / omega / 64
                t = dt * np.arange(4 * 64 + 1)
                eps0 = 1.0e-8
                reaction = HARMONIC.algorithmic_reaction(eps0 * np.sin(omega * t), G, TAU, dt, 1.0, 0.5, 0.0, integrator)
                m_iso, _ = HARMONIC.axial_modulus(1.0, 0.5)
                storage, loss = HARMONIC.harmonic_fit(t[1:], reaction[1:], eps0, omega, TAU, m_iso)
                expected = HARMONIC.discrete_complex_modulus(omega, G, TAU, dt, integrator)
                self.assertLess(abs(storage + 1j * loss - expected) / abs(expected), 1.0e-6)

    def test_transfer_function_tends_to_the_complex_modulus(self):
        exact = HARMONIC.complex_modulus(10.0, G, TAU)
        coarse = HARMONIC.discrete_complex_modulus(10.0, G, TAU, 1.0e-3)
        fine = HARMONIC.discrete_complex_modulus(10.0, G, TAU, 5.0e-4)
        self.assertAlmostEqual(abs(coarse - exact) / abs(fine - exact), 4.0, delta=0.1)


class BarModeOracleTests(unittest.TestCase):
    material = {"c10": 1.0, "c01": 0.5, "bulk_modulus": 2.0, "density": 1.0}

    def test_mesh_wave_number_error_is_second_order(self):
        # Consistent mass on linear elements: k2_h / k2 = 1 + (k h)^2 / 12 + O(h^4).
        k2 = BAR.wave_number_squared(1.0)
        for elements in (64, 256):
            kh = np.pi / elements
            self.assertAlmostEqual((BAR.wave_number_squared(1.0, elements) / k2 - 1.0) / (kh * kh / 12.0), 1.0, delta=1.0e-2)

    def test_elastic_limit_is_the_undamped_mode(self):
        # Relaxation times far beyond the period: the instantaneous modulus, no damping.
        mode = BAR.modal_parameters(self.material, "0.3,0.2", "1e9,2e9", BAR.wave_number_squared(1.0))
        self.assertAlmostEqual(mode["omega_d"], np.pi * np.sqrt(2.0 + 4.0), places=6)
        self.assertLess(mode["decay"], 1.0e-7)

    def test_newmark_converges_to_the_exact_mode(self):
        k2 = BAR.wave_number_squared(1.0, 64)
        errors = []
        for dt in (0.01, 0.005, 0.0025):
            steps = round(3.2 / dt)
            t = dt * np.arange(steps + 1)
            q = BAR.newmark_amplitude(steps, dt, self.material, "0.3,0.2", "0.05,0.3", k2, 1.0)
            exact = BAR.amplitude(t, self.material, "0.3,0.2", "0.05,0.3", k2, 1.0)
            errors.append(np.linalg.norm(q - exact) / np.linalg.norm(exact))
        self.assertGreater(np.log2(errors[0] / errors[1]), 1.95)
        self.assertGreater(np.log2(errors[1] / errors[2]), 1.95)


if __name__ == "__main__":
    unittest.main()
