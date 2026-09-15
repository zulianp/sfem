import unittest
from pathlib import Path
import sys

import numpy as np


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))

from common.affine import affine_checks  # noqa: E402
from common.geometry import box_mesh, rectangle_mesh  # noqa: E402
from common.hyperelastic import response  # noqa: E402
from common.mechanics import element_kinematics  # noqa: E402


MATERIAL = {"mu": 2.0, "lambda": 3.0, "c1": 1.25, "c2": 0.75, "kappa": 5.0}


def finite_difference_energy_gradient(operator, deformation, step=1.0e-7):
    gradient = np.empty_like(deformation)
    for i in range(deformation.shape[0]):
        for j in range(deformation.shape[1]):
            increment = np.zeros_like(deformation)
            increment[i, j] = step
            plus = response(operator, deformation + increment, MATERIAL)[0]
            minus = response(operator, deformation - increment, MATERIAL)[0]
            gradient[i, j] = (plus - minus) / (2.0 * step)
    return gradient


class HyperelasticOracleTests(unittest.TestCase):
    def test_first_piola_is_energy_gradient(self):
        deformations = (
            np.asarray(((1.08, 0.07), (-0.02, 0.96))),
            np.asarray(((1.08, 0.07, -0.03), (-0.02, 0.96, 0.04), (0.01, -0.05, 1.03))),
        )
        operators = (
            "GeneratedNeoHookeanOgden",
            "GeneratedModifiedMooneyRivlin",
            "GeneratedSaintVenantKirchhoff",
        )
        for deformation in deformations:
            for operator in operators:
                with self.subTest(dimension=len(deformation), operator=operator):
                    _, first_piola = response(operator, deformation, MATERIAL)
                    finite_difference = finite_difference_energy_gradient(operator, deformation)
                    np.testing.assert_allclose(first_piola, finite_difference, rtol=2.0e-7, atol=2.0e-8)

    def test_mu_and_lambda_perturbations_fail_physical_checks(self):
        mesh = rectangle_mesh(1.0, 1.0, 4, 4, "QUAD4")
        deformation = np.diag((1.1, 0.97))
        displacement = mesh.points @ (deformation - np.eye(2)).T
        baseline_energy, baseline_stress = response("GeneratedNeoHookeanOgden", deformation, MATERIAL)
        boundary = np.flatnonzero(
            np.isclose(mesh.points[:, 0], 0.0)
            | np.isclose(mesh.points[:, 0], 1.0)
            | np.isclose(mesh.points[:, 1], 0.0)
            | np.isclose(mesh.points[:, 1], 1.0)
        )
        interior = np.setdiff1d(np.arange(mesh.n_points), boundary)
        reaction_nodes = np.flatnonzero(np.isclose(mesh.points[:, 0], 1.0))
        tolerances = {
            "mode_displacement_relative_l2": 1.0e-9,
            "mode_free_residual_normalized": 1.0e-9,
            "mode_energy_relative": 1.0e-8,
            "mode_reaction_relative_l2": 1.0e-8,
        }
        for parameter in ("mu", "lambda"):
            perturbed = dict(MATERIAL)
            perturbed[parameter] *= 1.1
            observed_energy, observed_stress = response("GeneratedNeoHookeanOgden", deformation, perturbed)
            reaction = np.zeros_like(displacement)
            reaction[reaction_nodes] = (observed_stress @ np.asarray((1.0, 0.0))) / len(reaction_nodes)
            checks, _ = affine_checks(
                "mode",
                mesh,
                interior,
                reaction_nodes,
                displacement,
                reaction,
                observed_energy,
                deformation,
                baseline_stress,
                baseline_energy,
                tolerances,
                {"type": "analytical", "reference": "test"},
            )
            failed = {check["name"] for check in checks if not check["passed"]}
            with self.subTest(parameter=parameter):
                self.assertIn("mode_energy_relative", failed)
                self.assertIn("mode_reaction_relative_l2", failed)


class AffineSensitivityTests(unittest.TestCase):
    def test_displacement_component_perturbation_fails_oracle(self):
        mesh = rectangle_mesh(1.0, 1.0, 4, 4, "TRI3")
        deformation = np.asarray(((1.03, 0.02), (0.0, 0.99)))
        displacement = mesh.points @ (deformation - np.eye(2)).T
        energy_density, stress = _linear_response(deformation)
        boundary = np.flatnonzero(
            np.isclose(mesh.points[:, 0], 0.0)
            | np.isclose(mesh.points[:, 0], 1.0)
            | np.isclose(mesh.points[:, 1], 0.0)
            | np.isclose(mesh.points[:, 1], 1.0)
        )
        interior = np.setdiff1d(np.arange(mesh.n_points), boundary)
        reaction_nodes = np.flatnonzero(np.isclose(mesh.points[:, 0], 1.0))
        reaction = np.zeros_like(displacement)
        reaction[reaction_nodes] = (stress @ np.asarray((1.0, 0.0))) / len(reaction_nodes)
        tolerances = {
            "mode_displacement_relative_l2": 1.0e-10,
            "mode_free_residual_normalized": 1.0e-10,
            "mode_energy_relative": 1.0e-9,
            "mode_reaction_relative_l2": 1.0e-9,
        }
        perturbed = displacement.copy()
        perturbed[interior[0], 1] += 1.0e-3
        checks, _ = affine_checks(
            "mode",
            mesh,
            interior,
            reaction_nodes,
            perturbed,
            reaction,
            energy_density,
            deformation,
            stress,
            energy_density,
            tolerances,
            {"type": "analytical", "reference": "test"},
        )
        displacement_check = next(check for check in checks if check["name"].endswith("displacement_relative_l2"))
        self.assertFalse(displacement_check["passed"])

    def test_inverted_tetrahedron_is_rejected(self):
        mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1, "TET4")
        inverted = type(mesh)(mesh.points.copy(), mesh.elements.copy(), mesh.element_type)
        inverted.elements[0, [0, 1]] = inverted.elements[0, [1, 0]]
        with self.assertRaisesRegex(ValueError, "non-positive geometry Jacobian"):
            element_kinematics(inverted, np.zeros_like(inverted.points))


def _linear_response(deformation):
    strain = 0.5 * (deformation + deformation.T) - np.eye(len(deformation))
    stress = 2.0 * MATERIAL["mu"] * strain + MATERIAL["lambda"] * np.trace(strain) * np.eye(len(deformation))
    return float(0.5 * np.sum(stress * strain)), stress


if __name__ == "__main__":
    unittest.main()
