#!/usr/bin/env python3

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import yaml


def shape(xi: float, eta: float):
    values = 0.25 * np.array(
        [
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ]
    )
    gradients = 0.25 * np.array(
        [
            [-(1 - eta), -(1 - xi)],
            [1 - eta, -(1 + xi)],
            [1 + eta, 1 + xi],
            [-(1 + eta), 1 - xi],
        ]
    )
    return values, gradients


def deformation(reference, current, xi: float, eta: float):
    values, gradients = shape(xi, eta)
    reference_jacobian = reference.T @ gradients
    current_jacobian = current.T @ gradients
    return values @ reference, current_jacobian @ np.linalg.inv(reference_jacobian)


def cauchy_stress(f2, c1: float, c2: float, kappa: float):
    f = np.eye(3)
    f[:2, :2] = f2
    jacobian = np.linalg.det(f)
    if jacobian <= 0 or not math.isfinite(jacobian):
        raise ValueError(f"non-positive deformation Jacobian: {jacobian}")
    b = f @ f.T
    i1 = np.trace(b)
    i2 = 0.5 * (i1 * i1 - np.sum(b * b))
    identity = np.eye(3)
    sigma = (
        2 * c1 * jacobian ** (-5.0 / 3.0) * (b - (i1 / 3.0) * identity)
        + 2
        * c2
        * jacobian ** (-7.0 / 3.0)
        * (i1 * b - b @ b - (2.0 * i2 / 3.0) * identity)
        + kappa * math.log(jacobian) / jacobian * identity
    )
    return sigma, jacobian


def final_component(solution: Path, component: int):
    paths = sorted(glob.glob(str(solution / "out" / f"disp.{component}.*.float64")))
    if not paths:
        raise FileNotFoundError(f"missing displacement component {component} in {solution / 'out'}")
    return np.fromfile(paths[-1], dtype=np.float64)


def error_check(name, observed, tolerance, units, oracle):
    return {
        "name": name,
        "oracle": oracle,
        "observed": float(observed),
        "expected": 0.0,
        "error": float(observed),
        "tolerance": float(tolerance),
        "units": units,
        "passed": bool(observed <= tolerance),
    }


def evaluate_level(mesh, solution, nr, ntheta, c1, c2, kappa):
    points = np.column_stack(
        [
            np.fromfile(mesh / "x.float32", dtype=np.float32).astype(np.float64),
            np.fromfile(mesh / "y.float32", dtype=np.float32).astype(np.float64),
        ]
    )
    elements = np.column_stack(
        [np.fromfile(mesh / f"i{local_node}.int32", dtype=np.int32) for local_node in range(4)]
    )
    if elements.shape != (nr * ntheta, 4):
        raise ValueError(f"invalid {nr} x {ntheta} pressure-vessel connectivity")
    displacement = np.column_stack([final_component(solution, 0), final_component(solution, 1)])
    if displacement.shape != points.shape:
        raise ValueError(f"displacement shape {displacement.shape} does not match mesh {points.shape}")
    current = points + displacement

    radial_profile = []
    ray = ntheta // 2
    for radial_cell in range(nr):
        samples = []
        for angular_cell, eta in ((ray - 1, 1.0), (ray, -1.0)):
            element = radial_cell + nr * angular_cell
            nodes = elements[element]
            location, f2 = deformation(points[nodes], current[nodes], 0.0, eta)
            sigma, jacobian = cauchy_stress(f2, c1, c2, kappa)
            er2 = location / np.linalg.norm(location)
            er = np.array([er2[0], er2[1], 0.0])
            et = np.array([-er2[1], er2[0], 0.0])
            samples.append((np.linalg.norm(location), er @ sigma @ er, et @ sigma @ et, jacobian))
        radial_profile.append(np.mean(samples, axis=0))
    radial_profile = np.asarray(radial_profile)

    all_jacobians = []
    gauss = 1.0 / math.sqrt(3.0)
    for nodes in elements:
        for xi in (-gauss, gauss):
            for eta in (-gauss, gauss):
                _, f2 = deformation(points[nodes], current[nodes], xi, eta)
                all_jacobians.append(np.linalg.det(f2))
    all_jacobians = np.asarray(all_jacobians)
    return radial_profile, all_jacobians


def profile_error(profile, oracle_profile, column):
    radii = profile[:, 0]
    mask = (radii >= oracle_profile[0, 0]) & (radii <= oracle_profile[-1, 0])
    if np.count_nonzero(mask) < 2:
        raise ValueError("profile has insufficient samples inside the published oracle support")
    expected = np.interp(radii[mask], oracle_profile[:, 0], oracle_profile[:, 1])
    differences = profile[mask, column] - expected
    return float(np.linalg.norm(differences) / np.linalg.norm(expected)), float(np.max(np.abs(differences)))


def common_grid_errors(profiles, radial_oracle, hoop_oracle):
    lower = max(
        *(profile[0, 0] for profile in profiles.values()),
        radial_oracle[0, 0],
        hoop_oracle[0, 0],
    )
    upper = min(
        *(profile[-1, 0] for profile in profiles.values()),
        radial_oracle[-1, 0],
        hoop_oracle[-1, 0],
    )
    if not upper > lower:
        raise ValueError("refinement profiles have no common published-radius support")
    radii = np.linspace(lower, upper, 32)
    errors = {}
    for column, oracle_profile, label in ((1, radial_oracle, "radial"), (2, hoop_oracle, "hoop")):
        expected = np.interp(radii, oracle_profile[:, 0], oracle_profile[:, 1])
        denominator = np.linalg.norm(expected)
        errors[label] = {
            level_id: float(
                np.linalg.norm(np.interp(radii, profile[:, 0], profile[:, column]) - expected) / denominator
            )
            for level_id, profile in profiles.items()
        }
    return radii, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare pressure-vessel stresses with published analytical curves")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    case_path = args.case.resolve()
    case_dir = case_path.parent
    config = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    tolerances = config["verification"]["tolerances"]
    oracle = config["oracle"]
    c1 = float(config["driver"]["environment"]["SFEM_C1"])
    c2 = float(config["driver"]["environment"]["SFEM_C2"])
    kappa = float(config["driver"]["environment"]["SFEM_KAPPA"])

    radial_oracle = np.loadtxt(case_dir / oracle["radial_stress"], delimiter=",", comments="#")
    hoop_oracle = np.loadtxt(case_dir / oracle["hoop_stress"], delimiter=",", comments="#")
    for table in (radial_oracle, hoop_oracle):
        if table.ndim != 2 or table.shape[1] != 2 or not np.all(np.isfinite(table)):
            raise ValueError("published oracle tables must contain finite radius/stress pairs")
        if np.any(np.diff(table[:, 0]) <= 0):
            raise ValueError("published oracle radii must be strictly increasing")

    profiles = {}
    jacobians = {}
    for level in config["refinements"]:
        level_id = level["id"]
        cells = int(level["cells"])
        profiles[level_id], jacobians[level_id] = evaluate_level(
            args.output / "mesh" / level_id,
            args.output / "solution" / level_id,
            cells,
            cells,
            c1,
            c2,
            kappa,
        )

    canonical_id = next(level["id"] for level in config["refinements"] if level["role"] == "canonical")
    radial_profile = profiles[canonical_id]
    radial_relative_l2, radial_max_abs = profile_error(radial_profile, radial_oracle, 1)
    hoop_relative_l2, hoop_max_abs = profile_error(radial_profile, hoop_oracle, 2)
    common_radii, common_errors = common_grid_errors(profiles, radial_oracle, hoop_oracle)
    ordered_ids = [level["id"] for level in config["refinements"]]

    analytical_source = {
        "type": "published_analytical_profile",
        "reference": config["source"]["data"],
    }
    checks = [
        error_check(
            "radial_stress_relative_l2",
            radial_relative_l2,
            tolerances["radial_stress_relative_l2"],
            "1",
            analytical_source,
        ),
        error_check(
            "radial_stress_max_abs_mpa",
            radial_max_abs,
            tolerances["radial_stress_max_abs_mpa"],
            "MPa",
            analytical_source,
        ),
        error_check(
            "hoop_stress_relative_l2",
            hoop_relative_l2,
            tolerances["hoop_stress_relative_l2"],
            "1",
            analytical_source,
        ),
        error_check(
            "hoop_stress_max_abs_mpa",
            hoop_max_abs,
            tolerances["hoop_stress_max_abs_mpa"],
            "MPa",
            analytical_source,
        ),
    ]

    for label in ("radial", "hoop"):
        level_errors = common_errors[label]
        deficit = max(
            0.0,
            *(level_errors[ordered_ids[i + 1]] - level_errors[ordered_ids[i]]
              for i in range(len(ordered_ids) - 1)),
        )
        checks.append(
            error_check(
                f"{label}_refinement_monotonic_deficit",
                deficit,
                tolerances[f"{label}_refinement_monotonic_deficit"],
                "1",
                analytical_source,
            )
        )

    minimum_jacobian = min(float(np.min(values)) for values in jacobians.values())
    jacobian_deficit = max(0.0, np.finfo(np.float64).eps - minimum_jacobian)
    checks.append({
        "name": "minimum_deformation_jacobian_deficit",
        "oracle": {"type": "analytical", "reference": "positive material volume"},
        "observed": minimum_jacobian,
        "expected": 0.0,
        "error": jacobian_deficit,
        "tolerance": float(tolerances["minimum_deformation_jacobian_deficit"]),
        "units": "1",
        "passed": bool(jacobian_deficit <= tolerances["minimum_deformation_jacobian_deficit"]),
    })

    profile_artifacts = {}
    for level_id, level_profile in profiles.items():
        radii = level_profile[:, 0]
        profile = np.column_stack(
            [
                radii,
                level_profile[:, 1],
                np.interp(radii, radial_oracle[:, 0], radial_oracle[:, 1]),
                level_profile[:, 2],
                np.interp(radii, hoop_oracle[:, 0], hoop_oracle[:, 1]),
                level_profile[:, 3],
            ]
        )
        path = args.output / ("stress_profile.csv" if level_id == canonical_id else f"stress_profile_{level_id}.csv")
        np.savetxt(
            path,
            profile,
            delimiter=",",
            header="radius,sfem_radial_mpa,oracle_radial_mpa,sfem_hoop_mpa,oracle_hoop_mpa,jacobian",
            comments="",
        )
        profile_artifacts[f"stress_profile_{level_id}"] = str(path)

    report = {
        "schema_version": 1,
        "case": config["id"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "diagnostics": {
            "canonical_level": canonical_id,
            "common_radius_grid_m": [float(value) for value in common_radii],
            "common_grid_relative_l2": common_errors,
            "deformation_jacobian_gauss_point_min": minimum_jacobian,
            "deformation_jacobian_gauss_point_max": max(float(np.max(values)) for values in jacobians.values()),
            "levels": {
                level_id: {
                    "radial_stress_relative_l2_at_native_radii": profile_error(level_profile, radial_oracle, 1)[0],
                    "hoop_stress_relative_l2_at_native_radii": profile_error(level_profile, hoop_oracle, 2)[0],
                    "minimum_deformation_jacobian": float(np.min(jacobians[level_id])),
                }
                for level_id, level_profile in profiles.items()
            },
        },
        "artifacts": profile_artifacts,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status} {check['name']}: error={check['error']:.6g}, tolerance={check['tolerance']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
