#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.affine import read_component_output
from common.convergence import fit_spatial_convergence
from common.mechanics import element_kinematics
from common.mesh import read_mesh
from oracle import displacement as analytical_displacement, polar_stress


def check(name, observed, error, tolerance, units, reference, expected=0.0):
    return {
        "name": name,
        "oracle": {"type": "analytical", "reference": reference},
        "observed": float(observed),
        "expected": float(expected),
        "error": float(error),
        "tolerance": float(tolerance),
        "units": units,
        "passed": bool(error <= tolerance),
    }


def numerical_polar_stress(kinematics, mu, lmbda):
    strain = kinematics.small_strain
    sigma = 2.0 * mu * strain + lmbda * np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(2)
    locations = kinematics.locations
    er = locations / np.linalg.norm(locations, axis=-1)[..., None]
    et = np.stack((-er[..., 1], er[..., 0]), axis=-1)
    return np.stack(
        (
            np.einsum("...i,...ij,...j->...", er, sigma, er),
            np.einsum("...i,...ij,...j->...", et, sigma, et),
            np.einsum("...i,...ij,...j->...", er, sigma, et),
        ), axis=-1,
    )


def evaluate_level(level, output, config):
    level_id = level["id"]
    mesh = read_mesh(output / "mesh" / level_id)
    displacement = read_component_output(
        output / "solution" / level_id, "x", 2, mesh.n_points,
    )
    a = float(config["geometry"]["hole_radius_m"])
    tension = float(config["loading"]["remote_tension_mpa"])
    mu = float(config["material"]["mu"])
    lmbda = float(config["material"]["lambda"])
    expected_displacement = analytical_displacement(mesh.points, a, tension, mu, lmbda)
    displacement_error = float(
        np.linalg.norm(displacement - expected_displacement) / np.linalg.norm(expected_displacement)
    )

    kinematics = element_kinematics(mesh, displacement)
    observed_stress = numerical_polar_stress(kinematics, mu, lmbda)
    expected_stress = polar_stress(kinematics.locations, a, tension)
    weights = kinematics.weights
    stress_error = float(np.sqrt(
        np.sum(weights[..., None] * (observed_stress - expected_stress) ** 2)
        / np.sum(weights[..., None] * expected_stress ** 2)
    ))

    radial_cells = int(level["radial_cells"])
    angular_cells = int(level["angular_cells"])
    elements_per_cell = 2 if mesh.element_type == "TRI3" else 1
    shape = (angular_cells, radial_cells, elements_per_cell, observed_stress.shape[1])
    hoop = observed_stress[..., 1].reshape(shape)
    radii = np.linalg.norm(kinematics.locations, axis=-1).reshape(shape)
    near_hole = [
        (float(np.mean(radii[-1, cell])), float(np.mean(hoop[-1, cell])))
        for cell in (0, 1)
    ]
    (r0, s0), (r1, s1) = near_hole
    if r1 <= r0:
        raise ValueError(f"{level_id}: radial stress-recovery samples are not ordered")
    hole_hoop = s0 + (a - r0) * (s1 - s0) / (r1 - r0)
    hole_peak_error = abs(hole_hoop / (3.0 * tension) - 1.0)

    ray = angular_cells // 2
    observed = observed_stress.reshape(angular_cells, radial_cells, elements_per_cell,
                                      observed_stress.shape[1], 3)
    expected = expected_stress.reshape(observed.shape)
    ray_radii = np.mean(radii[ray], axis=(1, 2))
    ray_observed = np.mean(observed[ray], axis=(1, 2))
    ray_expected = np.mean(expected[ray], axis=(1, 2))
    path = output / f"stress_profile_{level_id}.csv"
    np.savetxt(
        path,
        np.column_stack((ray_radii, ray_observed, ray_expected)),
        delimiter=",",
        header=("radius_m,sfem_radial_mpa,sfem_hoop_mpa,sfem_shear_mpa,"
                "oracle_radial_mpa,oracle_hoop_mpa,oracle_shear_mpa"),
        comments="",
    )
    return {
        "displacement_relative_l2": displacement_error,
        "stress_relative_l2": stress_error,
        "hole_peak_hoop_mpa": float(hole_hoop),
        "hole_peak_hoop_relative_error": float(hole_peak_error),
        "mesh_point_count": mesh.n_points,
        "mesh_element_count": mesh.n_elements,
        "profile": str(path),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare static Kirsch fields with the analytical solution")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    levels = config["refinements"]
    results = {level["id"]: evaluate_level(level, args.output, config) for level in levels}
    ordered = [results[level["id"]] for level in levels]
    displacement_errors = [item["displacement_relative_l2"] for item in ordered]
    stress_errors = [item["stress_relative_l2"] for item in ordered]
    mesh_sizes = [1.0 / level["radial_cells"] for level in levels]
    fit = fit_spatial_convergence(mesh_sizes, displacement_errors)
    tolerance = config["verification"]["tolerances"]
    reference = "oracle.py: Kirsch plane-strain displacement and polar stress"
    checks = [
        check("finest_displacement_relative_l2", displacement_errors[-1], displacement_errors[-1],
              tolerance["finest_displacement_relative_l2"], "1", reference),
        check("finest_stress_relative_l2", stress_errors[-1], stress_errors[-1],
              tolerance["finest_stress_relative_l2"], "1", reference),
        check("hole_peak_hoop_relative_error", ordered[-1]["hole_peak_hoop_mpa"],
              ordered[-1]["hole_peak_hoop_relative_error"],
              tolerance["hole_peak_hoop_relative_error"], "1", reference,
              expected=3.0 * float(config["loading"]["remote_tension_mpa"])),
        check("displacement_monotonic_deficit", displacement_errors[-1],
              max(0.0, *(displacement_errors[i + 1] - displacement_errors[i]
                         for i in range(len(levels) - 1))),
              tolerance["displacement_monotonic_deficit"], "1", reference),
        check("stress_monotonic_deficit", stress_errors[-1],
              max(0.0, *(stress_errors[i + 1] - stress_errors[i]
                         for i in range(len(levels) - 1))),
              tolerance["stress_monotonic_deficit"], "1", reference),
        check("displacement_rate_deficit", fit.rate, max(0.0, 1.7 - fit.rate),
              tolerance["displacement_rate_deficit"], "1", reference, expected=1.7),
    ]
    report = {
        "schema_version": 1,
        "case": config["id"],
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "diagnostics": {
            "levels": {level["id"]: results[level["id"]] for level in levels},
            "displacement_convergence": fit.as_dict(),
            "geometry": yaml.safe_load((args.output / "mesh" / "geometry_diagnostics.yaml").read_text()),
        },
        "artifacts": {f"stress_profile_{level['id']}": results[level["id"]]["profile"] for level in levels},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}: "
              f"error={item['error']:.6g}, tolerance={item['tolerance']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
