#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.mesh import read_mesh
from common.metrics import relative_l2_error
from common.raw import read_raw
from common.reporting import build_verification_report, make_check, write_verification_report
from common.transient import TIME_LEVELS, read_field_history
from oracle import mode_coefficients, small_strain_exponential, smooth_ramp, solve_response


def main():
    parser = argparse.ArgumentParser(description="Verify 3D finite-strain Kelvin-Voigt creep")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    material = config["material"]
    tolerances = config["verification"]["tolerances"]
    mode = config["selected_variant"]["resolution"]["mode"]
    component = 0 if mode == "axial" else 1
    duration = float(config["time"]["duration"])
    time_steps = np.asarray(config["time"]["time_steps"], dtype=np.float64)
    traction = float(config["loading"]["traction"])
    ramp_time = float(config["loading"]["ramp_time"])
    mesh = read_mesh(args.output / "mesh")
    right = read_raw(args.output / "mesh" / "sets" / "right.int32.raw", dtype=np.int32).astype(np.int64)
    oracle = {"type": "analytical_ode", "reference": "finite_strain_kv_creep_3d/oracle.py"}

    errors = []
    level_data = {}
    for name, dt in zip(TIME_LEVELS, time_steps):
        expected_times = np.linspace(0.0, duration, round(duration / dt) + 1)
        times, displacement = read_field_history(
            args.output / "solution" / name / "out", "disp", 3, mesh.n_points, expected_times
        )
        _, reaction = read_field_history(
            args.output / "solution" / name / "out", "material_reaction", 3, mesh.n_points, expected_times
        )
        observed = np.mean(displacement[:, right, component], axis=1)
        expected = solve_response(times, mode, material, traction, ramp_time)
        observed_reaction = np.sum(reaction[:, right, component], axis=1)
        expected_reaction = traction * smooth_ramp(times, ramp_time)
        error = relative_l2_error(observed, expected)
        errors.append(max(error, np.finfo(np.float64).tiny))
        level_data[name] = (times, observed, expected, observed_reaction, expected_reaction)

    coarse = level_data["coarse"][1]
    medium = level_data["medium"][1]
    fine = level_data["fine"][1]
    coarse_medium = relative_l2_error(coarse, medium[::2])
    medium_fine = relative_l2_error(medium, fine[::2])
    temporal_rate = float(np.log(coarse_medium / medium_fine) / np.log(2.0))
    times, response, expected, reaction, expected_reaction = level_data["fine"]
    history_error = relative_l2_error(response, expected)
    final_error = abs(response[-1] - expected[-1]) / max(abs(expected[-1]), 1.0e-14)
    reaction_error = relative_l2_error(reaction[1:], expected_reaction[1:])

    stiffness, viscosity, _ = mode_coefficients(mode, material)
    small_times = np.linspace(0.0, 1.0, 101)
    nonlinear = solve_response(small_times, mode, material, 1.0e-5, 1.0e-12)
    exponential = small_strain_exponential(small_times, stiffness, viscosity, 1.0e-5)
    small_error = relative_l2_error(nonlinear, exponential)
    checks = [
        make_check("response_history_relative_l2", history_error, 0, history_error,
                   tolerances["response_history_relative_l2"], "1", oracle),
        make_check("final_response_relative", response[-1], expected[-1], final_error,
                   tolerances["final_response_relative"], "1", oracle),
        make_check("reaction_history_relative_l2", reaction_error, 0, reaction_error,
                   tolerances["reaction_history_relative_l2"], "1", oracle),
        make_check("temporal_rate_deficit", temporal_rate, 1.8, max(0.0, 1.8 - temporal_rate),
                   tolerances["temporal_rate_deficit"], "1", oracle),
        make_check("small_strain_limit_relative_l2", small_error, 0, small_error,
                   tolerances["small_strain_limit_relative_l2"], "1", oracle),
    ]
    history_path = args.output / "history.csv"
    np.savetxt(history_path, np.column_stack((times, response, expected, reaction, expected_reaction)),
               delimiter=",", header="time,response,oracle_response,reaction,oracle_reaction", comments="")
    report = build_verification_report(
        config["id"], checks,
        diagnostics={"mode": mode, "time_steps": time_steps.tolist(), "history_errors": errors,
                     "temporal_self_errors": [coarse_medium, medium_fine], "temporal_rate": temporal_rate},
        artifacts={"history": history_path},
    )
    write_verification_report(args.report, report, tolerances=tolerances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
