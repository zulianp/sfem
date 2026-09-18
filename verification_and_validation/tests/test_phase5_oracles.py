import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SUITE_DIR = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, SUITE_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CREEP_2D = load_module("phase5_creep_2d", "finite_strain_kv_creep_2d/oracle.py")
CREEP_3D = load_module("phase5_creep_3d", "finite_strain_kv_creep_3d/oracle.py")
MODE = load_module("phase5_mode", "linear_kv_damped_mode_3d/oracle.py")
PRONY = load_module("phase5_prony", "prony_relaxation_3d/oracle.py")


class Phase5OracleTests(unittest.TestCase):
    def test_finite_strain_creep_recovers_small_strain_exponential(self):
        times = np.linspace(0.0, 1.0, 101)
        material_2d = {"mu": 2.0, "lambda": 3.0, "eta_s": 0.4, "eta_b": 0.2}
        traction = 1.0e-7
        observed_2d = CREEP_2D.solve_axial(times, material_2d, traction, 1.0e-10)
        expected_2d = CREEP_2D.small_strain_exponential(
            times, 6.0 * material_2d["mu"] + material_2d["lambda"], 0.6, traction
        )
        np.testing.assert_allclose(observed_2d - 1.0, expected_2d - 1.0, rtol=2.0e-6, atol=2.0e-12)

        material_3d = {"mu": 2.0, "lambda": 3.0, "eta_s": 0.3, "eta_b": 0.2}
        stiffness, viscosity, _ = CREEP_3D.mode_coefficients("axial", material_3d)
        observed_3d = CREEP_3D.solve_response(times, "axial", material_3d, traction, 1.0e-10)
        expected_3d = CREEP_3D.small_strain_exponential(times, stiffness, viscosity, traction)
        np.testing.assert_allclose(observed_3d, expected_3d, rtol=2.0e-6, atol=2.0e-12)

    def test_damped_mode_satisfies_initial_conditions_and_ode(self):
        material = {"shear_stiffness": 4.0, "bulk_modulus": 8.0, "damping": 0.05, "density": 1.0}
        parameters = MODE.modal_parameters(material, 1.0)
        q0 = 0.01
        v0 = 0.02
        h = 1.0e-6
        values = MODE.amplitude(np.asarray((-h, 0.0, h)), q0, v0, parameters["decay"], parameters["omega_d"])
        velocity = (values[2] - values[0]) / (2.0 * h)
        acceleration = (values[2] - 2.0 * values[1] + values[0]) / h**2
        expected_acceleration = -2.0 * parameters["decay"] * v0 - parameters["omega_0"] ** 2 * q0
        self.assertAlmostEqual(q0, values[1], places=14)
        self.assertAlmostEqual(v0, velocity, places=10)
        self.assertAlmostEqual(expected_acceleration, acceleration, places=5)

    def test_prony_limits_and_wlf_reduced_time(self):
        g = np.asarray((0.25, 0.2))
        tau = np.asarray((0.2, 0.8))
        np.testing.assert_allclose(PRONY.relaxation_factor((0.0,), g, tau), (1.0,), atol=1.0e-15)
        np.testing.assert_allclose(PRONY.relaxation_factor((100.0,), g, tau), (1.0 - np.sum(g),), atol=1.0e-14)

        wlf = {"C1": 2.0, "C2": 100.0, "T_ref": 20.0}
        hot_time = np.linspace(0.0, 2.0, 21)
        shift = PRONY.wlf_shift(30.0, wlf["C1"], wlf["C2"], wlf["T_ref"])
        hot = PRONY.relaxation_factor(hot_time, g, tau, temperature=30.0, wlf=wlf)
        reference = PRONY.relaxation_factor(hot_time * shift, g, tau, temperature=20.0, wlf=wlf)
        np.testing.assert_allclose(hot, reference, rtol=2.0e-15, atol=2.0e-15)


class TransientHistoryTests(unittest.TestCase):
    def test_missing_intermediate_field_state_is_rejected(self):
        import sys

        sys.path.insert(0, str(SUITE_DIR))
        from common.transient import read_field_history

        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            np.savetxt(folder / "time.txt", (0.0, 1.0))
            np.asarray((0.0, 0.0), dtype=np.float64).tofile(folder / "disp.0.0.float64")
            with self.assertRaisesRegex(ValueError, "contains 1 disp states.*expected 2"):
                read_field_history(folder, "disp", 1, 2, (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
