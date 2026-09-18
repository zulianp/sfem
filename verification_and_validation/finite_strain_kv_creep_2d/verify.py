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
from oracle import small_strain_exponential, smooth_ramp, solve_axial


def main():
    parser = argparse.ArgumentParser(description="Verify 2D finite-strain Kelvin-Voigt creep")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    material = config["material"]
    tolerances = config["verification"]["tolerances"]
    duration = float(config["time"]["duration"])
    time_steps = np.asarray(config["time"]["time_steps"], dtype=np.float64)
    traction = float(config["loading"]["traction"])
    ramp_time = float(config["loading"]["ramp_time"])
    mesh = read_mesh(args.output / "mesh")
    right = read_raw(args.output / "mesh" / "sets" / "right.int32.raw", dtype=np.int32).astype(np.int64)
    oracle = {"type": "analytical_ode", "reference": "finite_strain_kv_creep_2d/oracle.py"}

    errors = []
    level_data = {}
    for name, dt in zip(TIME_LEVELS, time_steps):
        expected_times = np.linspace(0.0, duration, round(duration / dt) + 1)
        times, displacement = read_field_history(
            args.output / "solution" / name / "out", "disp", 2, mesh.n_points, expected_times
        )
        _, reaction = read_field_history(
            args.output / "solution" / name / "out", "material_reaction", 2, mesh.n_points, expected_times
        )
        observed_stretch = 1.0 + np.mean(displacement[:, right, 0], axis=1)
        expected_stretch = solve_axial(times, material, traction, ramp_time)
        observed_reaction = np.sum(reaction[:, right, 0], axis=1)
        expected_reaction = traction * smooth_ramp(times, ramp_time)
        error = relative_l2_error(observed_stretch - 1.0, expected_stretch - 1.0)
        errors.append(max(error, np.finfo(np.float64).tiny))
        level_data[name] = (times, observed_stretch, expected_stretch, observed_reaction, expected_reaction)

    coarse = level_data["coarse"][1] - 1.0
    medium = level_data["medium"][1] - 1.0
    fine = level_data["fine"][1] - 1.0
    coarse_medium = relative_l2_error(coarse, medium[::2])
    medium_fine = relative_l2_error(medium, fine[::2])
    temporal_rate = float(np.log(coarse_medium / medium_fine) / np.log(2.0))
    times, stretch, expected_stretch, reaction, expected_reaction = level_data["fine"]
    stretch_error = relative_l2_error(stretch - 1.0, expected_stretch - 1.0)
    final_error = abs(stretch[-1] - expected_stretch[-1]) / max(abs(expected_stretch[-1] - 1.0), 1.0e-14)
    reaction_error = relative_l2_error(reaction[1:], expected_reaction[1:])

    tiny_traction = 1.0e-5
    small_times = np.linspace(0.0, 1.0, 101)
    tiny_material = dict(material)
    nonlinear = solve_axial(small_times, tiny_material, tiny_traction, 1.0e-12)
    stiffness = 6.0 * float(material["mu"]) + float(material["lambda"])
    viscosity = float(material["eta_s"]) + float(material["eta_b"])
    exponential = small_strain_exponential(small_times, stiffness, viscosity, tiny_traction)
    small_error = relative_l2_error(nonlinear - 1.0, exponential - 1.0)

    checks = [
        make_check("stretch_history_relative_l2", stretch_error, 0, stretch_error,
                   tolerances["stretch_history_relative_l2"], "1", oracle),
        make_check("final_stretch_relative", stretch[-1], expected_stretch[-1], final_error,
                   tolerances["final_stretch_relative"], "1", oracle),
        make_check("reaction_history_relative_l2", reaction_error, 0, reaction_error,
                   tolerances["reaction_history_relative_l2"], "1", oracle),
        make_check("temporal_rate_deficit", temporal_rate, 1.8, max(0.0, 1.8 - temporal_rate),
                   tolerances["temporal_rate_deficit"], "1", oracle),
        make_check("small_strain_limit_relative_l2", small_error, 0, small_error,
                   tolerances["small_strain_limit_relative_l2"], "1", oracle),
    ]
    history_path = args.output / "history.csv"
    np.savetxt(
        history_path,
        np.column_stack((times, stretch, expected_stretch, reaction, expected_reaction)),
        delimiter=",",
        header="time,stretch,oracle_stretch,reaction,oracle_reaction",
        comments="",
    )
    report = build_verification_report(
        config["id"], checks,
        diagnostics={"time_steps": time_steps.tolist(), "history_errors": errors,
                     "temporal_self_errors": [coarse_medium, medium_fine], "temporal_rate": temporal_rate},
        artifacts={"history": history_path},
    )
    write_verification_report(args.report, report, tolerances=tolerances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
