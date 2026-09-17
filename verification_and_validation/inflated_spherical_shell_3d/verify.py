#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.mechanics import element_kinematics
from common.mesh import Mesh, read_mesh
from common.sets import read_sideset, side_nodes, surface_geometry
from oracle import radial_fields, solve_radial_shell


def check(name, observed, expected, error, tolerance, units, reference):
    return {
        "name": name,
        "oracle": {"type": "semi_analytical", "reference": reference},
        "observed": float(observed),
        "expected": float(expected),
        "error": float(error),
        "tolerance": float(tolerance),
        "units": units,
        "passed": bool(error <= tolerance),
    }


def component_history(folder, component, node_count):
    paths = sorted(glob.glob(str(folder / f"disp.{component}.*.float64")))
    if not paths:
        raise FileNotFoundError(f"missing displacement history for component {component} in {folder}")
    arrays = [np.fromfile(path, dtype=np.float64) for path in paths]
    if any(len(array) != node_count or not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"incomplete or non-finite displacement component {component} in {folder}")
    return np.asarray(arrays)


def displacement_history(solution, node_count, expected_times):
    folder = solution / "out"
    times = np.loadtxt(folder / "time.txt", dtype=np.float64, ndmin=1)
    if times.shape != expected_times.shape or not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"incomplete or incorrect time history in {folder / 'time.txt'}")
    components = [component_history(folder, axis, node_count) for axis in range(3)]
    if any(len(component) != len(times) for component in components):
        raise ValueError(f"displacement history length does not match times in {folder}")
    return np.stack(components, axis=-1)


def cavity_volume(mesh, sideset, displacement):
    deformed_mesh = Mesh(mesh.points + displacement, mesh.elements, mesh.element_type)
    geometry = surface_geometry(deformed_mesh, sideset)
    return -float(np.sum(np.einsum("ij,ij->i", geometry.centroids, geometry.area_vectors))) / 3.0


def full_cauchy_stress(deformation, operator, material):
    deformation = np.asarray(deformation, dtype=np.float64)
    jacobian = np.linalg.det(deformation)
    if np.any(~np.isfinite(jacobian)) or np.any(jacobian <= 0.0):
        raise ValueError("non-positive or non-finite finite-strain quadrature Jacobian")
    inverse_transpose = np.swapaxes(np.linalg.inv(deformation), -1, -2)
    log_j = np.log(jacobian)
    if operator == "GeneratedNeoHookeanOgden":
        mu = float(material["mu"])
        lmbda = float(material["lambda"])
        nominal = mu * deformation + (lmbda * log_j - mu)[..., None, None] * inverse_transpose
    elif operator == "GeneratedModifiedMooneyRivlin":
        c1 = float(material["c1"])
        c2 = float(material["c2"])
        kappa = float(material["kappa"])
        right_cauchy_green = np.matmul(np.swapaxes(deformation, -1, -2), deformation)
        invariant_1 = np.trace(right_cauchy_green, axis1=-2, axis2=-1)
        invariant_2 = 0.5 * (
            invariant_1 ** 2 - np.sum(right_cauchy_green * right_cauchy_green, axis=(-2, -1))
        )
        nominal = 2.0 * c1 * jacobian[..., None, None] ** (-2.0 / 3.0) * (
            deformation - invariant_1[..., None, None] * inverse_transpose / 3.0
        )
        nominal += 2.0 * c2 * jacobian[..., None, None] ** (-4.0 / 3.0) * (
            invariant_1[..., None, None] * deformation
            - np.matmul(deformation, right_cauchy_green)
            - 2.0 * invariant_2[..., None, None] * inverse_transpose / 3.0
        )
        nominal += (kappa * log_j)[..., None, None] * inverse_transpose
    else:
        raise ValueError(f"unsupported finite-strain stress extraction: {operator}")
    return np.matmul(nominal, np.swapaxes(deformation, -1, -2)) / jacobian[..., None, None]


def evaluate_level(level, output, config, oracle_solutions, times):
    level_id = level["id"]
    mesh = read_mesh(output / "mesh" / level_id)
    history = displacement_history(output / "solution" / level_id, mesh.n_points, times)
    inner = read_sideset(output / "mesh" / level_id / "sets" / "inner")
    inner_nodes = np.unique(side_nodes(mesh, inner).ravel())
    a = float(config["geometry"]["inner_radius_m"])
    b = float(config["geometry"]["outer_radius_m"])
    operator = config["selected_variant"]["operator"]
    material = config["material"]
    reference_directions = mesh.points[inner_nodes] / a
    inner_radial_displacements = np.einsum("tni,ni->tn", history[:, inner_nodes], reference_directions)
    inner_radius = a + np.mean(inner_radial_displacements, axis=1)
    oracle_inner_radius = np.asarray(
        [radial_fields(solution, np.asarray([a]), operator, material)[0, 0]
         if solution is not None else a for solution in oracle_solutions],
    )
    oracle_inner_displacement = oracle_inner_radius[1:] - a
    observed_inner_displacement = inner_radius[1:] - a
    displacement_error = float(
        np.linalg.norm(observed_inner_displacement - oracle_inner_displacement)
        / np.linalg.norm(oracle_inner_displacement)
    )
    cavity_volumes = np.asarray([cavity_volume(mesh, inner, state) for state in history])
    if np.any(cavity_volumes <= 0.0) or not np.all(np.isfinite(cavity_volumes)):
        raise ValueError(f"{level_id}: non-positive or non-finite cavity volume")
    observed_volume = cavity_volumes - cavity_volumes[0]
    oracle_volume = np.pi / 6.0 * (oracle_inner_radius ** 3 - a ** 3)
    volume_error = float(
        np.linalg.norm(observed_volume[1:] - oracle_volume[1:]) / np.linalg.norm(oracle_volume[1:])
    )
    curve = output / f"pressure_volume_{level_id}.csv"
    pressure = float(config["loading"]["inner_pressure_mpa"]) * times / float(config["loading"]["ramp_duration_s"])
    np.savetxt(
        curve,
        np.column_stack((times, pressure, inner_radius, oracle_inner_radius,
                         observed_volume, oracle_volume)),
        delimiter=",",
        header="time_s,pressure_mpa,sfem_inner_radius_m,oracle_inner_radius_m,sfem_delta_volume_m3,oracle_delta_volume_m3",
        comments="",
    )

    final = history[-1]
    kinematics = element_kinematics(mesh, final)
    stress = full_cauchy_stress(kinematics.deformation_gradient, operator, material)
    locations = kinematics.locations
    radius = np.linalg.norm(locations, axis=-1)
    direction = locations / radius[..., None]
    radial_stress = np.einsum("...i,...ij,...j->...", direction, stress, direction)
    hoop_stress = 0.5 * (np.trace(stress, axis1=-2, axis2=-1) - radial_stress)
    expected_stress = radial_fields(oracle_solutions[-1], radius, operator, material)
    weights = kinematics.weights
    local_radial_error = float(np.sqrt(
        np.sum(weights * (radial_stress - expected_stress[..., 4]) ** 2)
        / np.sum(weights * expected_stress[..., 4] ** 2)
    ))
    local_hoop_error = float(np.sqrt(
        np.sum(weights * (hoop_stress - expected_stress[..., 5]) ** 2)
        / np.sum(weights * expected_stress[..., 5] ** 2)
    ))

    sample_radius = radius.ravel()
    sample_radial = radial_stress.ravel()
    sample_hoop = hoop_stress.ravel()
    sample_expected = expected_stress.reshape(-1, 6)
    sample_weights = weights.ravel()
    boundaries = np.linspace(a, b, int(level["radial_cells"]) + 1)
    samples = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        ids = np.flatnonzero((sample_radius >= lower) & (sample_radius < upper))
        if not len(ids):
            raise ValueError(f"{level_id}: missing stress sample in radial layer")
        shell_weights = sample_weights[ids]
        total_weight = np.sum(shell_weights)
        samples.append((
            np.sum(shell_weights * sample_radius[ids]) / total_weight,
            np.sum(shell_weights * sample_radial[ids]) / total_weight,
            np.sum(shell_weights * sample_hoop[ids]) / total_weight,
            np.sum(shell_weights * sample_expected[ids, 4]) / total_weight,
            np.sum(shell_weights * sample_expected[ids, 5]) / total_weight,
            total_weight,
        ))
    samples = np.asarray(samples)
    radial_error = float(np.sqrt(
        np.sum(samples[:, 5] * (samples[:, 1] - samples[:, 3]) ** 2)
        / np.sum(samples[:, 5] * samples[:, 3] ** 2)
    ))
    hoop_error = float(np.sqrt(
        np.sum(samples[:, 5] * (samples[:, 2] - samples[:, 4]) ** 2)
        / np.sum(samples[:, 5] * samples[:, 4] ** 2)
    ))
    profile = output / f"stress_profile_{level_id}.csv"
    np.savetxt(
        profile, samples, delimiter=",",
        header="reference_radius_m,sfem_radial_mpa,sfem_hoop_mpa,oracle_radial_mpa,oracle_hoop_mpa,shell_volume_m3",
        comments="",
    )
    return {
        "inner_displacement_relative_l2": displacement_error,
        "pressure_volume_relative_l2": volume_error,
        "radial_stress_relative_l2": radial_error,
        "hoop_stress_relative_l2": hoop_error,
        "local_quadrature_radial_stress_relative_l2": local_radial_error,
        "local_quadrature_hoop_stress_relative_l2": local_hoop_error,
        "minimum_jacobian": float(np.min(kinematics.deformation_jacobian)),
        "maximum_jacobian": float(np.max(kinematics.deformation_jacobian)),
        "inner_radius_history_m": inner_radius.tolist(),
        "oracle_inner_radius_history_m": oracle_inner_radius.tolist(),
        "cavity_volume_history_m3": cavity_volumes.tolist(),
        "mesh_point_count": mesh.n_points,
        "mesh_element_count": mesh.n_elements,
        "pressure_volume_curve": str(curve),
        "stress_profile": str(profile),
    }


def monotonic_deficit(values):
    return max(0.0, *(values[i + 1] - values[i] for i in range(len(values) - 1)))


def main():
    parser = argparse.ArgumentParser(description="Verify finite-strain pressured sphere against radial BVP")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    steps = int(config["loading"]["steps"])
    duration = float(config["loading"]["ramp_duration_s"])
    dt = float(config["loading"]["dt_s"])
    if abs(steps * dt - duration) > 1.0e-12:
        raise ValueError("time-step count and ramp duration disagree")
    times = np.linspace(0.0, duration, steps + 1)
    a = float(config["geometry"]["inner_radius_m"])
    b = float(config["geometry"]["outer_radius_m"])
    final_pressure = float(config["loading"]["inner_pressure_mpa"])
    operator = config["selected_variant"]["operator"]
    material = config["material"]
    oracle_solutions = [None] + [
        solve_radial_shell(operator, a, b, final_pressure * time / duration, material)
        for time in times[1:]
    ]
    levels = config["refinements"]
    results = {level["id"]: evaluate_level(level, args.output, config, oracle_solutions, times)
               for level in levels}
    ordered = [results[level["id"]] for level in levels]
    tolerance = config["verification"]["tolerances"]
    reference = "oracle.py: finite-strain radial equilibrium BVP, P_r(A)=-p lambda_t(A)^2"
    displacement_errors = [item["inner_displacement_relative_l2"] for item in ordered]
    radial_errors = [item["radial_stress_relative_l2"] for item in ordered]
    hoop_errors = [item["hoop_stress_relative_l2"] for item in ordered]
    minimum_j = min(item["minimum_jacobian"] for item in ordered)
    checks = [
        check("finest_inner_displacement_relative_l2", displacement_errors[-1], 0.0,
              displacement_errors[-1], tolerance["finest_inner_displacement_relative_l2"], "1", reference),
        check("finest_radial_stress_relative_l2", radial_errors[-1], 0.0,
              radial_errors[-1], tolerance["finest_radial_stress_relative_l2"], "1", reference),
        check("finest_hoop_stress_relative_l2", hoop_errors[-1], 0.0,
              hoop_errors[-1], tolerance["finest_hoop_stress_relative_l2"], "1", reference),
        check("finest_pressure_volume_relative_l2", ordered[-1]["pressure_volume_relative_l2"], 0.0,
              ordered[-1]["pressure_volume_relative_l2"],
              tolerance["finest_pressure_volume_relative_l2"], "1", reference),
        check("displacement_monotonic_deficit", displacement_errors[-1], 0.0,
              monotonic_deficit(displacement_errors), tolerance["displacement_monotonic_deficit"], "1", reference),
        check("radial_stress_monotonic_deficit", radial_errors[-1], 0.0,
              monotonic_deficit(radial_errors), tolerance["radial_stress_monotonic_deficit"], "1", reference),
        check("hoop_stress_monotonic_deficit", hoop_errors[-1], 0.0,
              monotonic_deficit(hoop_errors), tolerance["hoop_stress_monotonic_deficit"], "1", reference),
        check("minimum_jacobian_deficit", minimum_j, 0.0,
              max(0.0, np.finfo(np.float64).eps - minimum_j),
              tolerance["minimum_jacobian_deficit"], "1", reference),
    ]
    report = {
        "schema_version": 1,
        "case": config["id"],
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "diagnostics": {
            "levels": results,
            "times_s": times.tolist(),
            "geometry": yaml.safe_load((args.output / "mesh" / "geometry_diagnostics.yaml").read_text()),
        },
        "artifacts": {
            **{f"pressure_volume_{level['id']}": results[level["id"]]["pressure_volume_curve"] for level in levels},
            **{f"stress_profile_{level['id']}": results[level["id"]]["stress_profile"] for level in levels},
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}: "
              f"error={item['error']:.6g}, tolerance={item['tolerance']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
