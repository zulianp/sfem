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
    mesh = args.output / "mesh"
    solution = args.output / "solution"

    points = np.column_stack(
        [
            np.fromfile(mesh / "x.float32", dtype=np.float32).astype(np.float64),
            np.fromfile(mesh / "y.float32", dtype=np.float32).astype(np.float64),
        ]
    )
    elements = np.column_stack(
        [np.fromfile(mesh / f"i{local_node}.int32", dtype=np.int32) for local_node in range(4)]
    )
    displacement = np.column_stack([final_component(solution, 0), final_component(solution, 1)])
    if displacement.shape != points.shape:
        raise ValueError(f"displacement shape {displacement.shape} does not match mesh {points.shape}")
    current = points + displacement

    nr = int(config["mesh"]["command"][config["mesh"]["command"].index("--nr") + 1])
    ntheta = int(config["mesh"]["command"][config["mesh"]["command"].index("--ntheta") + 1])
    c1 = float(config["driver"]["environment"]["SFEM_C1"])
    c2 = float(config["driver"]["environment"]["SFEM_C2"])
    kappa = float(config["driver"]["environment"]["SFEM_KAPPA"])

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

    radial_oracle = np.loadtxt(case_dir / oracle["radial_stress"], delimiter=",", comments="#")
    hoop_oracle = np.loadtxt(case_dir / oracle["hoop_stress"], delimiter=",", comments="#")
    radii = radial_profile[:, 0]
    radial_mask = (radii >= radial_oracle[0, 0]) & (radii <= radial_oracle[-1, 0])
    hoop_mask = (radii >= hoop_oracle[0, 0]) & (radii <= hoop_oracle[-1, 0])
    expected_radial = np.interp(radii[radial_mask], radial_oracle[:, 0], radial_oracle[:, 1])
    expected_hoop = np.interp(radii[hoop_mask], hoop_oracle[:, 0], hoop_oracle[:, 1])
    radial_error = radial_profile[radial_mask, 1] - expected_radial
    hoop_error = radial_profile[hoop_mask, 2] - expected_hoop
    radial_relative_l2 = np.linalg.norm(radial_error) / np.linalg.norm(expected_radial)
    hoop_relative_l2 = np.linalg.norm(hoop_error) / np.linalg.norm(expected_hoop)

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
            np.max(np.abs(radial_error)),
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
            np.max(np.abs(hoop_error)),
            tolerances["hoop_stress_max_abs_mpa"],
            "MPa",
            analytical_source,
        ),
    ]

    expected_radial_all = np.interp(radii, radial_oracle[:, 0], radial_oracle[:, 1])
    expected_hoop_all = np.interp(radii, hoop_oracle[:, 0], hoop_oracle[:, 1])
    profile = np.column_stack(
        [radii, radial_profile[:, 1], expected_radial_all, radial_profile[:, 2], expected_hoop_all, radial_profile[:, 3]]
    )
    np.savetxt(
        args.output / "stress_profile.csv",
        profile,
        delimiter=",",
        header="radius,sfem_radial_mpa,oracle_radial_mpa,sfem_hoop_mpa,oracle_hoop_mpa,jacobian",
        comments="",
    )
    report = {
        "schema_version": 1,
        "case": config["id"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "diagnostics": {
            "deformation_jacobian_gauss_point_min": float(np.min(all_jacobians)),
            "deformation_jacobian_gauss_point_max": float(np.max(all_jacobians)),
        },
        "artifacts": {"stress_profile": str(args.output / "stress_profile.csv")},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status} {check['name']}: error={check['error']:.6g}, tolerance={check['tolerance']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
