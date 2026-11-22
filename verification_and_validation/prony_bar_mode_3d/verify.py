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
from oracle import amplitude, check_series, modal_parameters, newmark_amplitude, wave_number_squared


def project_mode(mesh, displacement, length):
    """The sine's amplitude in the axial displacement, and the field outside it (axial residual and
    lateral motion) relative to the largest displacement of the history."""
    shape = np.sin(np.pi * mesh.points[:, 0] / float(length))
    modal = displacement[:, :, 0] @ shape / float(np.dot(shape, shape))
    residual = displacement[:, :, 0] - modal[:, None] * shape[None, :]
    outside = np.sqrt(np.linalg.norm(residual, axis=1) ** 2 + np.linalg.norm(displacement[:, :, 1:], axis=(1, 2)) ** 2)
    return modal, outside / float(np.max(np.linalg.norm(displacement, axis=(1, 2))))


def fit_mode(times, observed, rates, expected):
    """delta and omega_d of q = exp(-delta t)(A cos + B sin)(omega_d t) + sum_i C_i exp(-r_i t), the
    relaxation rates r_i fixed, the amplitudes solved linearly for each (delta, omega_d)."""
    def design(parameters):
        delta, omega = parameters
        columns = [np.exp(-delta * times) * np.cos(omega * times), np.exp(-delta * times) * np.sin(omega * times)]
        columns += [np.exp(-r * times) for r in rates]
        return np.column_stack(columns)

    def residual(parameters):
        A = design(parameters)
        coefficients, *_ = np.linalg.lstsq(A, observed, rcond=None)
        return A @ coefficients - observed

    start = [expected["decay"], expected["omega_d"]]
    result = least_squares(residual, start, x_scale=np.abs(start), xtol=1.0e-15, ftol=1.0e-15, gtol=1.0e-15)
    return float(result.x[0]), float(result.x[1])


def main():
    parser = argparse.ArgumentParser(description="Verify the Prony bar's free mode")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    resolution = config["selected_variant"]["resolution"]
    tolerances = config["verification"]["tolerances"]
    material = config["material"]
    length = float(config["geometry"]["length"])
    v0 = float(config["initial_conditions"]["modal_velocity"])
    duration = float(config["time"]["duration"])
    time_steps = np.asarray(config["time"]["time_steps"], dtype=np.float64)
    integrator = str(resolution["integrator"])
    g, tau = check_series(str(resolution["prony_g"]), str(resolution["prony_tau"]))
    k2_mesh = wave_number_squared(length, int(resolution["nx"]))
    k2_continuum = wave_number_squared(length)
    mesh_mode = modal_parameters(material, g, tau, k2_mesh)
    continuum_mode = modal_parameters(material, g, tau, k2_continuum)
    oracle = {"type": "semi-analytical", "reference": "prony_bar_mode_3d/oracle.py"}

    levels, discrete_errors, history_errors, off_mode = {}, [], [], []
    for name, dt in zip(TIME_LEVELS, time_steps):
        mesh = read_mesh(args.output / "solution" / name / "mesh")
        steps = round(duration / dt)
        expected_times = dt * np.arange(steps + 1)
        times, displacement = read_field_history(
            args.output / "solution" / name / "out", "disp", 3, mesh.n_points, expected_times)
        observed, outside = project_mode(mesh, displacement, length)
        off_mode.append(float(np.max(outside)))
        # The fully discrete scheme on the mode: Newmark and the material's recursion.
        discrete = newmark_amplitude(steps, dt, material, g, tau, k2_mesh, v0, integrator)
        discrete_errors.append(float(np.max(np.abs(observed - discrete)) / np.max(np.abs(discrete))))
        # The mesh's mode exact in time.
        exact = amplitude(times, material, g, tau, k2_mesh, v0)
        history_errors.append(relative_l2_error(observed, exact))
        levels[name] = (times, observed, exact)

    rates = [float(np.log2(history_errors[k] / history_errors[k + 1])) for k in range(len(history_errors) - 1)]
    times, observed, exact = levels[TIME_LEVELS[-1]]
    decay, omega = fit_mode(times, observed, mesh_mode["relaxation_rates"], mesh_mode)
    measured = {
        "discrete_amplitude_max_relative_error": max(discrete_errors),
        "amplitude_history_relative_l2": history_errors[-1],
        "frequency_relative_error": abs(omega - mesh_mode["omega_d"]) / mesh_mode["omega_d"],
        "decay_relative_error": abs(decay - mesh_mode["decay"]) / mesh_mode["decay"],
        "continuum_frequency_relative_error": abs(omega - continuum_mode["omega_d"]) / continuum_mode["omega_d"],
        "continuum_decay_relative_error": abs(decay - continuum_mode["decay"]) / continuum_mode["decay"],
        "off_mode_fraction_max": max(off_mode),
        "temporal_rate_deficit": max(0.0, 1.8 - min(rates)),
    }
    checks = []
    for key in tolerances:
        value = float(measured[key])
        if key == "temporal_rate_deficit":
            checks.append(make_check(key, min(rates), 1.8, value, tolerances[key], "1", oracle))
        else:
            checks.append(make_check(key, value, 0, value, tolerances[key], "1", oracle))

    history_path = args.output / "modal_history.csv"
    np.savetxt(history_path, np.column_stack((times, observed, exact)), delimiter=",",
               header="time,amplitude,oracle_amplitude", comments="")
    report = build_verification_report(
        config["id"], checks,
        diagnostics={"integrator": integrator, "mesh_mode": mesh_mode, "continuum_mode": continuum_mode,
                     "fitted_decay": decay, "fitted_omega_d": omega,
                     # The log decrement per period, about pi tan(delta) at the mode's frequency.
                     "log_decrement": 2.0 * np.pi * decay / omega,
                     "discrete_errors": discrete_errors, "history_errors": history_errors, "temporal_rates": rates,
                     "off_mode_fraction": off_mode, "measured": {k: float(v) for k, v in measured.items()}},
        artifacts={"modal_history": history_path},
    )
    write_verification_report(args.report, report, tolerances=tolerances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
