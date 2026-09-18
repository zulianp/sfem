import importlib.util
import unittest
from pathlib import Path

import numpy as np


SUITE_DIR = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, SUITE_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACTIVE = load_module("phase6_active", "active_strain_homogeneous_3d/oracle.py")
MULTIBLOCK = load_module("phase6_multiblock", "linear_multiblock_patch_3d/oracle.py")


class ActiveStrainOracleTests(unittest.TestCase):
    material = {"mu": 2.0, "lambda": 5.0}

    def test_compatible_deformation_is_stress_free(self):
        active = ACTIVE.active_gradient()
        self.assertAlmostEqual(1.0, np.linalg.det(active), places=14)
        for operator in (
            "NeoHookeanOgdenActiveStrainPacked",
            "MooneyRivlinActiveStrainPacked",
        ):
            with self.subTest(operator=operator):
                _, stress = ACTIVE.response(operator, active, active, self.material)
                np.testing.assert_allclose(stress, 0.0, atol=2.0e-15)

    def test_first_piola_stress_is_energy_derivative(self):
        active = ACTIVE.active_gradient()
        deformation = np.asarray(
            ((1.04, 0.03, -0.01), (0.02, 0.98, 0.04), (0.01, -0.02, 1.02)),
            dtype=np.float64,
        )
        step = 1.0e-7
        for operator in (
            "NeoHookeanOgdenActiveStrainPacked",
            "MooneyRivlinActiveStrainPacked",
        ):
            with self.subTest(operator=operator):
                _, stress = ACTIVE.response(operator, deformation, active, self.material)
                numerical = np.empty((3, 3), dtype=np.float64)
                for i in range(3):
                    for j in range(3):
                        perturbation = np.zeros((3, 3), dtype=np.float64)
                        perturbation[i, j] = step
                        plus, _ = ACTIVE.response(
                            operator, deformation + perturbation, active, self.material
                        )
                        minus, _ = ACTIVE.response(
                            operator, deformation - perturbation, active, self.material
                        )
                        numerical[i, j] = (plus - minus) / (2.0 * step)
                np.testing.assert_allclose(numerical, stress, rtol=2.0e-8, atol=2.0e-9)


class MultiblockOracleTests(unittest.TestCase):
    def test_piecewise_extension_has_continuous_interface_traction(self):
        material = {
            "left_mu": 2.0,
            "left_lambda": 3.0,
            "right_mu": 5.0,
            "right_lambda": 7.0,
        }
        values = MULTIBLOCK.response(material)
        left_stress = (material["left_lambda"] + 2.0 * material["left_mu"]) * values[
            "left_strain"
        ]
        right_stress = (material["right_lambda"] + 2.0 * material["right_mu"]) * values[
            "right_strain"
        ]
        self.assertAlmostEqual(values["stress"], left_stress, places=14)
        self.assertAlmostEqual(values["stress"], right_stress, places=14)

        points = np.asarray(((0.5, 0.0, 0.0), (1.0, 0.0, 0.0)))
        displacement = MULTIBLOCK.displacement(points, material)[:, 0]
        self.assertAlmostEqual(0.5 * values["left_strain"], displacement[0], places=14)
        self.assertAlmostEqual(values["total_displacement"], displacement[1], places=14)
        self.assertAlmostEqual(
            0.5 * values["stress"] * values["total_displacement"],
            values["energy"],
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
