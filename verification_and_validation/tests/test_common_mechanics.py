import sys
import unittest
from pathlib import Path

import numpy as np


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))

from common.geometry import box_mesh, rectangle_mesh  # noqa: E402
from common.mechanics import (  # noqa: E402
    boundary_resultant_from_stress,
    cauchy_from_first_piola,
    element_kinematics,
    first_piola_from_cauchy,
    integrate_boundary_traction,
    integrate_strain_energy_density,
    pressure_resultant,
    small_strain_energy_density,
)
from common.sets import select_boundary_axis  # noqa: E402


class KinematicsTests(unittest.TestCase):
    def test_affine_gradient_is_exact_for_supported_elements(self):
        cases = (
            rectangle_mesh(2.0, 1.0, 2, 2, "TRI3"),
            rectangle_mesh(2.0, 1.0, 2, 2, "QUAD4"),
            box_mesh(2.0, 1.0, 3.0, 2, 1, 2, "TET4"),
            box_mesh(2.0, 1.0, 3.0, 2, 1, 2, "HEX8"),
        )
        for mesh in cases:
            with self.subTest(element=mesh.element_type):
                if mesh.dimension == 2:
                    gradient = np.asarray(((0.2, -0.1), (0.3, 0.05)))
                    translation = np.asarray((0.4, -0.2))
                    expected_volume = 2.0
                else:
                    gradient = np.asarray(((0.2, -0.1, 0.03), (0.3, 0.05, -0.02), (0.1, 0.04, -0.1)))
                    translation = np.asarray((0.4, -0.2, 0.1))
                    expected_volume = 6.0
                displacement = mesh.points @ gradient.T + translation
                kinematics = element_kinematics(mesh, displacement)

                expected_gradient = np.broadcast_to(gradient, kinematics.displacement_gradient.shape)
                np.testing.assert_allclose(kinematics.displacement_gradient, expected_gradient, atol=2.0e-15)
                np.testing.assert_allclose(
                    kinematics.small_strain,
                    np.broadcast_to(0.5 * (gradient + gradient.T), kinematics.small_strain.shape),
                    atol=2.0e-15,
                )
                self.assertAlmostEqual(expected_volume, float(np.sum(kinematics.weights)))
                self.assertTrue(np.all(kinematics.deformation_jacobian > 0.0))

    def test_first_piola_and_cauchy_round_trip(self):
        deformation = np.asarray(((1.2, 0.1, 0.0), (0.0, 0.9, 0.2), (0.0, 0.0, 1.1)))
        cauchy = np.asarray(((4.0, 0.3, 0.1), (0.3, 2.0, -0.2), (0.1, -0.2, 1.0)))
        first_piola = first_piola_from_cauchy(cauchy, deformation)
        recovered = cauchy_from_first_piola(first_piola, deformation)
        np.testing.assert_allclose(cauchy, recovered, atol=1.0e-14)

    def test_energy_density_and_volume_integration(self):
        mesh = rectangle_mesh(2.0, 1.0, 2, 1)
        gradient = np.asarray(((0.1, 0.0), (0.0, -0.05)))
        displacement = mesh.points @ gradient.T
        kinematics = element_kinematics(mesh, displacement)
        stress = np.broadcast_to(np.asarray(((3.0, 0.0), (0.0, 2.0))), kinematics.small_strain.shape)
        density = small_strain_energy_density(stress, kinematics.small_strain)

        self.assertAlmostEqual(0.1, density[0, 0])
        self.assertAlmostEqual(0.2, integrate_strain_energy_density(density, kinematics.weights))


class ResultantTests(unittest.TestCase):
    def setUp(self):
        self.mesh = box_mesh(2.0, 1.0, 3.0, 2, 1, 3)
        self.right = select_boundary_axis(self.mesh, 0, 2.0)

    def test_constant_stress_resultant(self):
        stress = np.diag((3.0, 2.0, 1.0))
        resultant = boundary_resultant_from_stress(self.mesh, self.right, stress)
        np.testing.assert_allclose(resultant, (9.0, 0.0, 0.0), atol=1.0e-14)

    def test_traction_and_pressure_resultants(self):
        traction = integrate_boundary_traction(self.mesh, self.right, np.asarray((2.0, -1.0, 0.5)))
        pressure = pressure_resultant(self.mesh, self.right, 4.0)
        np.testing.assert_allclose(traction, (6.0, -3.0, 1.5), atol=1.0e-14)
        np.testing.assert_allclose(pressure, (-12.0, 0.0, 0.0), atol=1.0e-14)


if __name__ == "__main__":
    unittest.main()
