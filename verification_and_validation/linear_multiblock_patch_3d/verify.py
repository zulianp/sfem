#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.affine import read_component_output
from common.mesh import read_mesh
from common.metrics import relative_l2_error
from common.raw import read_raw
from common.reporting import build_verification_report, make_check, write_verification_report
from oracle import displacement, response


def main():
    parser = argparse.ArgumentParser(description="Verify the two-material linear patch")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    tolerances = config["verification"]["tolerances"]
    mesh_root = args.output / "mesh"
    mesh = read_mesh(mesh_root / "oracle_mesh")
    interior = read_raw(mesh_root / "sets/interior.int32.raw", np.int32).astype(np.int64)
    right = read_raw(mesh_root / "sets/right.int32.raw", np.int32).astype(np.int64)
    solution = args.output / "solution"
    observed_displacement = read_component_output(solution, "x", 3, mesh.n_points)
    reaction = read_component_output(solution, "material_reaction", 3, mesh.n_points)
    quantities = yaml.safe_load((solution / "quantities.yaml").read_text(encoding="utf-8"))
    observed_energy = float(quantities["material_objective"])

    material = config["material"]
    exact = response(material)
    expected_displacement = displacement(mesh.points, material)
    expected_resultant = np.asarray((exact["stress"], 0.0, 0.0))
    observed_resultant = np.sum(reaction[right], axis=0)
    displacement_error = relative_l2_error(observed_displacement, expected_displacement)
    residual_error = float(np.linalg.norm(reaction[interior])) / exact["stress"]
    energy_error = abs(observed_energy - exact["energy"]) / abs(exact["energy"])
    reaction_error = relative_l2_error(observed_resultant, expected_resultant)
    oracle = {"type": "analytical", "reference": "linear_multiblock_patch_3d/oracle.py"}
    values = {
        "displacement_relative_l2": (displacement_error, 0.0, displacement_error),
        "free_residual_normalized": (residual_error, 0.0, residual_error),
        "energy_relative": (observed_energy, exact["energy"], energy_error),
        "reaction_relative_l2": (reaction_error, 0.0, reaction_error),
    }
    checks = [
        make_check(name, observed, expected, error, tolerances[name], "1", oracle)
        for name, (observed, expected, error) in values.items()
    ]
    diagnostics = {
        "left_strain": exact["left_strain"],
        "right_strain": exact["right_strain"],
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
