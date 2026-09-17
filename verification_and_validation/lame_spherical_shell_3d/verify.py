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
from common.sets import read_sideset, side_nodes
from oracle import displacement as analytical_displacement, octant_pressure_resultant, polar_stress


def check(name, observed, expected, error, tolerance, units, reference):
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


def evaluate_level(level, output, config, geometry):
    level_id = level["id"]
    mesh = read_mesh(output / "mesh" / level_id)
    solution = output / "solution" / level_id
    observed_displacement = read_component_output(solution, "x", 3, mesh.n_points)
    reaction = read_component_output(solution, "material_reaction", 3, mesh.n_points)
    a = float(config["geometry"]["inner_radius_m"])
    b = float(config["geometry"]["outer_radius_m"])
    pressure = float(config["loading"]["inner_pressure_mpa"])
    mu = float(config["material"]["mu"])
    lmbda = float(config["material"]["lambda"])

    expected_displacement = analytical_displacement(mesh.points, a, b, pressure, mu, lmbda)
    displacement_error = float(
        np.linalg.norm(observed_displacement - expected_displacement) / np.linalg.norm(expected_displacement)
    )
    kinematics = element_kinematics(mesh, observed_displacement)
    strain = kinematics.small_strain
    stress = 2.0 * mu * strain + lmbda * np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
    locations = kinematics.locations
    radius = np.linalg.norm(locations, axis=-1)
    direction = locations / radius[..., None]
    radial = np.einsum("...i,...ij,...j->...", direction, stress, direction)
    hoop = 0.5 * (np.trace(stress, axis1=-2, axis2=-1) - radial)
    expected = polar_stress(radius, a, b, pressure)
    weights = kinematics.weights
    radial_error = float(np.sqrt(
        np.sum(weights * (radial - expected[..., 0]) ** 2)
        / np.sum(weights * expected[..., 0] ** 2)
    ))
    hoop_error = float(np.sqrt(
        np.sum(weights * (hoop - expected[..., 1]) ** 2)
        / np.sum(weights * expected[..., 1] ** 2)
    ))

    inner_sides = read_sideset(output / "mesh" / level_id / "sets" / "inner")
    inner_nodes = np.unique(side_nodes(mesh, inner_sides).ravel())
    material_inner_resultant = np.sum(reaction[inner_nodes], axis=0)
    expected_resultant = octant_pressure_resultant(a, pressure)
    reaction_resultant_error = float(
        np.linalg.norm(material_inner_resultant - expected_resultant) / np.linalg.norm(expected_resultant)
    )
    generated_resultant = np.asarray(geometry["applied_pressure_resultant_mpa_m2"])
    applied_resultant_error = float(
        np.linalg.norm(generated_resultant - expected_resultant) / np.linalg.norm(expected_resultant)
    )

    ray = np.full(3, 1.0 / np.sqrt(3.0))
    sample_directions = direction.reshape(-1, 3)
    sample_radii = radius.ravel()
    sample_radial = radial.ravel()
    sample_hoop = hoop.ravel()
    sample_expected = expected.reshape(-1, 2)
    boundaries = np.linspace(a, b, int(level["radial_cells"]) + 1)
    samples = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        ids = np.flatnonzero((sample_radii >= lower) & (sample_radii < upper))
        if not len(ids):
            raise ValueError(f"{level_id}: missing stress sample in radial layer")
        index = ids[np.argmax(sample_directions[ids] @ ray)]
        samples.append((sample_radii[index], sample_radial[index], sample_hoop[index],
                        sample_expected[index, 0], sample_expected[index, 1]))
    path = output / f"stress_profile_{level_id}.csv"
    np.savetxt(
        path, np.asarray(samples), delimiter=",",
        header="radius_m,sfem_radial_mpa,sfem_hoop_mpa,oracle_radial_mpa,oracle_hoop_mpa",
        comments="",
    )
    return {
        "displacement_relative_l2": displacement_error,
        "radial_stress_relative_l2": radial_error,
        "hoop_stress_relative_l2": hoop_error,
        "material_inner_resultant_mpa_m2": material_inner_resultant.tolist(),
        "material_inner_resultant_relative_l2": reaction_resultant_error,
        "applied_resultant_relative_l2": applied_resultant_error,
        "mesh_point_count": mesh.n_points,
        "mesh_element_count": mesh.n_elements,
        "profile": str(path),
    }


def monotonic_deficit(values):
    return max(0.0, *(values[i + 1] - values[i] for i in range(len(values) - 1)))


def main():
    parser = argparse.ArgumentParser(description="Compare pressure-loaded spherical octant with Lame solution")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    geometry = yaml.safe_load((args.output / "mesh" / "geometry_diagnostics.yaml").read_text())
    levels = config["refinements"]
    results = {
        level["id"]: evaluate_level(level, args.output, config, geometry[level["id"]])
        for level in levels
    }
    ordered = [results[level["id"]] for level in levels]
    displacement_errors = [item["displacement_relative_l2"] for item in ordered]
    radial_errors = [item["radial_stress_relative_l2"] for item in ordered]
    hoop_errors = [item["hoop_stress_relative_l2"] for item in ordered]
    fit = fit_spatial_convergence([1.0 / level["radial_cells"] for level in levels], displacement_errors)
    tolerance = config["verification"]["tolerances"]
    reference = "oracle.py: Lame radial equilibrium, displacement, and octant resultant"
    checks = [
        check("finest_displacement_relative_l2", displacement_errors[-1], 0.0,
              displacement_errors[-1], tolerance["finest_displacement_relative_l2"], "1", reference),
        check("finest_radial_stress_relative_l2", radial_errors[-1], 0.0,
              radial_errors[-1], tolerance["finest_radial_stress_relative_l2"], "1", reference),
        check("finest_hoop_stress_relative_l2", hoop_errors[-1], 0.0,
              hoop_errors[-1], tolerance["finest_hoop_stress_relative_l2"], "1", reference),
        check("finest_pressure_resultant_relative_l2", ordered[-1]["material_inner_resultant_relative_l2"],
              0.0, ordered[-1]["material_inner_resultant_relative_l2"],
              tolerance["finest_pressure_resultant_relative_l2"], "1", reference),
        check("finest_applied_pressure_resultant_relative_l2", ordered[-1]["applied_resultant_relative_l2"],
              0.0, ordered[-1]["applied_resultant_relative_l2"],
              tolerance["finest_applied_pressure_resultant_relative_l2"], "1", reference),
        check("displacement_monotonic_deficit", displacement_errors[-1], 0.0,
              monotonic_deficit(displacement_errors), tolerance["displacement_monotonic_deficit"], "1", reference),
        check("radial_stress_monotonic_deficit", radial_errors[-1], 0.0,
              monotonic_deficit(radial_errors), tolerance["radial_stress_monotonic_deficit"], "1", reference),
        check("hoop_stress_monotonic_deficit", hoop_errors[-1], 0.0,
              monotonic_deficit(hoop_errors), tolerance["hoop_stress_monotonic_deficit"], "1", reference),
        check("displacement_rate_deficit", fit.rate, 1.6, max(0.0, 1.6 - fit.rate),
              tolerance["displacement_rate_deficit"], "1", reference),
    ]
    report = {
        "schema_version": 1,
        "case": config["id"],
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "diagnostics": {
            "levels": results,
            "displacement_convergence": fit.as_dict(),
            "geometry": geometry,
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
