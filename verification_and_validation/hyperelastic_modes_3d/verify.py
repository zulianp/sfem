#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.affine import affine_checks, load_affine_mesh, read_mode_solution
from common.mechanics import element_kinematics
from common.reporting import build_verification_report, write_verification_report
from oracle import deformation_gradients, response


def main():
    parser = argparse.ArgumentParser(description="Verify three-dimensional homogeneous hyperelastic modes")
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
    reference_kinematics = element_kinematics(mesh, np.zeros_like(mesh.points))
    volume = float(np.sum(reference_kinematics.weights))
    oracle = {"type": "analytical", "reference": "hyperelastic_modes_3d/oracle.py"}

    checks = []
    diagnostics = {"reference_volume": volume, "operator": operator, "modes": {}}
    for mode, deformation in deformation_gradients().items():
        displacement, reaction, objective = read_mode_solution(
            args.output / "solution", mode, "hyperelastic", mesh.dimension, mesh.n_points
        )
        kinematics = element_kinematics(mesh, displacement)
        energy_density, first_piola = response(operator, deformation, material)
        mode_checks, mode_diagnostics = affine_checks(
            mode,
            mesh,
            artifacts["interior_nodes"],
            artifacts["reaction_nodes"],
            displacement,
            reaction,
            objective,
            deformation,
            first_piola,
            energy_density * volume,
            tolerances,
            oracle,
            deformation_jacobians=kinematics.deformation_jacobian,
        )
        checks.extend(mode_checks)
        diagnostics["modes"][mode] = mode_diagnostics

    report = build_verification_report(config["id"], checks, diagnostics=diagnostics)
    write_verification_report(args.report, report, tolerances=tolerances)
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['error']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
