#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common.mesh import read_mesh
from common.metrics import relative_l2_error
from common.reporting import build_verification_report, make_check, write_verification_report
from common.transient import TIME_LEVELS, read_field_history
from oracle import amplitude, modal_parameters


def project_mode(mesh, displacement, length):
    shape = np.sin(np.pi * mesh.points[:, 0] / float(length))
    denominator = float(np.dot(shape, shape))
    modal_amplitude = displacement[:, :, 0] @ shape / denominator
    projected = modal_amplitude[:, None] * shape[None, :]
    residual = displacement[:, :, 0] - projected
    off_mode = np.linalg.norm(residual, axis=1) / np.maximum(np.linalg.norm(displacement[:, :, 0], axis=1), 1.0e-14)
    transverse = np.linalg.norm(displacement[:, :, 1:], axis=(1, 2))
    total = np.maximum(np.linalg.norm(displacement, axis=(1, 2)), 1.0e-14)
    return modal_amplitude, np.maximum(off_mode, transverse / total)


def fit_parameters(times, observed, q0, v0, expected):
    def residual(parameters):
        return amplitude(times, q0, v0, parameters[0], parameters[1]) - observed

    result = least_squares(
        residual,
        [expected["decay"], expected["omega_d"]],
        bounds=([0.0, 0.5 * expected["omega_d"]], [5.0 * expected["decay"], 1.5 * expected["omega_d"]]),
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
    )
    return float(result.x[0]), float(result.x[1])


def main():
    parser = argparse.ArgumentParser(description="Verify the 3D Kelvin-Voigt damped mode")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    tolerances = config["verification"]["tolerances"]
    material = config["material"]
    length = float(config["geometry"]["length"])
    q0 = float(config["initial_conditions"]["modal_amplitude"])
    v0 = float(config["initial_conditions"]["modal_velocity"])
    duration = float(config["time"]["duration"])
    time_steps = np.asarray(config["time"]["time_steps"], dtype=np.float64)
    parameters = modal_parameters(material, length)
    oracle = {"type": "analytical", "reference": "linear_kv_damped_mode_3d/oracle.py"}

    errors = []
    levels = {}
    for name, dt in zip(TIME_LEVELS, time_steps):
        mesh = read_mesh(args.output / "solution" / name / "mesh")
        expected_times = np.linspace(0.0, duration, round(duration / dt) + 1)
        times, displacement = read_field_history(
            args.output / "solution" / name / "out", "disp", 3, mesh.n_points, expected_times
        )
        observed, off_mode = project_mode(mesh, displacement, length)
        expected = amplitude(times, q0, v0, parameters["decay"], parameters["omega_d"])
        error = relative_l2_error(observed, expected)
        errors.append(max(error, np.finfo(np.float64).tiny))
        levels[name] = (times, observed, expected, off_mode)

    coarse = levels["coarse"][1]
    medium = levels["medium"][1]
    fine = levels["fine"][1]
    coarse_medium = relative_l2_error(coarse, medium[::2])
    medium_fine = relative_l2_error(medium, fine[::2])
    temporal_rate = float(np.log(coarse_medium / medium_fine) / np.log(2.0))
    times, observed, expected, off_mode = levels["fine"]
    amplitude_error = relative_l2_error(observed, expected)
    max_error = float(np.max(np.abs(observed - expected))) / max(abs(q0), abs(v0 / parameters["omega_d"]), 1.0e-14)
    fitted_decay, fitted_frequency = fit_parameters(times, observed, q0, v0, parameters)
    decay_error = abs(fitted_decay - parameters["decay"]) / parameters["decay"]
    frequency_error = abs(fitted_frequency - parameters["omega_d"]) / parameters["omega_d"]

    checks = [
        make_check("amplitude_history_relative_l2", amplitude_error, 0, amplitude_error,
                   tolerances["amplitude_history_relative_l2"], "1", oracle),
        make_check("maximum_normalized_amplitude_error", max_error, 0, max_error,
                   tolerances["maximum_normalized_amplitude_error"], "1", oracle),
        make_check("frequency_relative_error", fitted_frequency, parameters["omega_d"], frequency_error,
                   tolerances["frequency_relative_error"], "1", oracle),
        make_check("decay_relative_error", fitted_decay, parameters["decay"], decay_error,
                   tolerances["decay_relative_error"], "1", oracle),
        make_check("temporal_rate_deficit", temporal_rate, 1.8, max(0.0, 1.8 - temporal_rate),
                   tolerances["temporal_rate_deficit"], "1", oracle),
    ]
    history_path = args.output / "modal_history.csv"
    np.savetxt(history_path, np.column_stack((times, observed, expected, off_mode)), delimiter=",",
               header="time,amplitude,oracle_amplitude,off_mode_fraction", comments="")
    report = build_verification_report(
        config["id"], checks,
        diagnostics={"parameters": parameters, "fitted_decay": fitted_decay,
                     "fitted_frequency": fitted_frequency, "maximum_off_mode_fraction": float(np.max(off_mode)),
                     "time_steps": time_steps.tolist(), "history_errors": errors,
                     "temporal_self_errors": [coarse_medium, medium_fine], "temporal_rate": temporal_rate},
        artifacts={"modal_history": history_path},
    )
    write_verification_report(args.report, report, tolerances=tolerances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
