#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.affine import load_affine_mesh, read_mode_solution
from common.mechanics import boundary_resultant_from_stress, element_kinematics
from common.metrics import relative_l2_error
from common.reporting import build_verification_report, make_check, write_verification_report
from common.sets import surface_geometry
from oracle import active_gradient, deformation_gradients, response


def main():
    parser = argparse.ArgumentParser(description="Verify homogeneous active strain")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    operator = config["selected_variant"]["operator"]
    material = config["material"]
    tolerances = config["verification"]["tolerances"]
    artifacts = load_affine_mesh(args.output / "mesh")
    mesh = artifacts["mesh"]
    volume = float(np.sum(element_kinematics(mesh, np.zeros_like(mesh.points)).weights))
    area = float(np.sum(surface_geometry(mesh, artifacts["reaction_sides"]).measures))
    oracle = {"type": "analytical", "reference": "active_strain_homogeneous_3d/oracle.py"}

    checks = []
    diagnostics = {
        "operator": operator,
        "reference_volume": volume,
        "active_gradient": active_gradient().tolist(),
        "modes": {},
    }
    for mode, deformation in deformation_gradients().items():
        displacement, reaction, objective = read_mode_solution(
            args.output / "solution", mode, "hyperelastic", mesh.dimension, mesh.n_points
        )
        energy_density, first_piola = response(operator, deformation, active_gradient(), material)
        expected_displacement = mesh.points @ (deformation - np.eye(3)).T
        expected_resultant = boundary_resultant_from_stress(mesh, artifacts["reaction_sides"], first_piola)
        observed_resultant = np.sum(reaction[artifacts["reaction_nodes"]], axis=0)
        expected_energy = energy_density * volume
        displacement_error = relative_l2_error(displacement, expected_displacement)
        residual_error = float(np.linalg.norm(reaction[artifacts["interior_nodes"]])) / max(
            float(material["mu"]) * area, 1.0e-14
        )
        energy_error = abs(objective - expected_energy) / max(abs(expected_energy), float(material["mu"]) * volume)
        reaction_error = float(np.linalg.norm(observed_resultant - expected_resultant)) / max(
            float(np.linalg.norm(expected_resultant)), float(material["mu"]) * area
        )
        values = {
            f"{mode}_displacement_relative_l2": (displacement_error, 0.0, displacement_error),
            f"{mode}_free_residual_normalized": (residual_error, 0.0, residual_error),
            f"{mode}_energy_relative": (objective, expected_energy, energy_error),
            f"{mode}_reaction_relative_l2": (reaction_error, 0.0, reaction_error),
        }
        for name, (observed, expected, error) in values.items():
            checks.append(make_check(name, observed, expected, error, tolerances[name], "1", oracle))
        diagnostics["modes"][mode] = {
            "observed_energy": objective,
            "expected_energy": expected_energy,
            "observed_reaction_resultant": observed_resultant.tolist(),
            "expected_reaction_resultant": expected_resultant.tolist(),
        }

    report = build_verification_report(config["id"], checks, diagnostics=diagnostics)
    write_verification_report(args.report, report, tolerances=tolerances)
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['error']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
