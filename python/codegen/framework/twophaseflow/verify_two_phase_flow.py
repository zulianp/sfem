#!/usr/bin/env python3
import argparse
import csv
import io
import math
import os
import shutil
import subprocess

import numpy as np


PARAMETERS = {
    "porosity": 0.2,
    "s_res": 0.1,
    "p_r": 1e5,
    "m": 2.0,
    "rho_w0": 1000.0,
    "kappa_t": 1e-9,
    "p_wr": 1e5,
    "m_c": 0.044,
    "z": 1.0,
    "r": 8.314462618,
    "temperature": 300.0,
    "mu_w": 1e-3,
    "mu_c": 1.5e-5,
    "c_kw1": 2.0,
    "c_ka1": 2.0,
    "c_ka2": 2.0,
    "permeability": 1e-12,
}

LOCAL_COORDINATES = np.array(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    )
)
GAUSS_POINTS = (0.21132486540518713, 0.7886751345948129)
GAUSS_WEIGHT = 0.5


def constitutive(pw, pc):
    p = PARAMETERS
    suction = pc - pw
    sw = p["s_res"] + (1.0 - p["s_res"]) * (
        1.0 + (suction / p["p_r"]) ** p["m"]
    ) ** (1.0 / p["m"] - 1.0)
    sc = 1.0 - sw
    se = (sw - p["s_res"]) / (1.0 - p["s_res"])
    rho_w = p["rho_w0"] * np.exp(p["kappa_t"] * (pw - p["p_wr"]))
    rho_c = p["m_c"] * pc / (p["z"] * p["r"] * p["temperature"])
    krw = np.sqrt(sw) * (
        1.0 - (1.0 - sw ** (1.0 / p["c_kw1"])) ** p["c_kw1"]
    ) ** 2
    krc = (1.0 - se) ** p["c_ka1"] * (1.0 - se ** p["c_ka2"])
    return sw, sc, rho_w, rho_c, krw, krc


def shape_data(qx, qy, qz, hx, hy, hz):
    q = np.asarray((qx, qy, qz))
    h = np.asarray((hx, hy, hz))
    shape = np.empty(8)
    gradient = np.empty((8, 3))
    for node, local in enumerate(LOCAL_COORDINATES):
        factors = np.where(local == 0.0, 1.0 - q, q)
        signs = np.where(local == 0.0, -1.0, 1.0)
        shape[node] = factors[0] * factors[1] * factors[2]
        gradient[node, 0] = (
            signs[0] * factors[1] * factors[2] / h[0]
        )
        gradient[node, 1] = (
            factors[0] * signs[1] * factors[2] / h[1]
        )
        gradient[node, 2] = (
            factors[0] * factors[1] * signs[2] / h[2]
        )
    return shape, gradient


def structured_hex8(nx=2, ny=1, nz=1):
    coordinates = np.empty(((nx + 1) * (ny + 1) * (nz + 1), 3))
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                node = ix + (nx + 1) * (iy + (ny + 1) * iz)
                coordinates[node] = (ix / nx, 0.5 * iy / ny, 0.5 * iz / nz)
    elements = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                node = lambda dx, dy, dz: (
                    ix + dx
                    + (nx + 1) * (iy + dy + (ny + 1) * (iz + dz))
                )
                elements.append(
                    (
                        node(0, 0, 0),
                        node(1, 0, 0),
                        node(1, 1, 0),
                        node(0, 1, 0),
                        node(0, 0, 1),
                        node(1, 0, 1),
                        node(1, 1, 1),
                        node(0, 1, 1),
                    )
                )
    return coordinates, np.asarray(elements, dtype=np.int64)


def residual(state, previous, dt, coordinates, elements):
    result = np.zeros_like(state)
    p = PARAMETERS
    for element in elements:
        xyz = coordinates[element]
        hx = xyz[:, 0].max() - xyz[:, 0].min()
        hy = xyz[:, 1].max() - xyz[:, 1].min()
        hz = xyz[:, 2].max() - xyz[:, 2].min()
        determinant = hx * hy * hz
        current_e = state.reshape(-1, 2)[element]
        previous_e = previous.reshape(-1, 2)[element]
        local = np.zeros((8, 2))
        for qz in GAUSS_POINTS:
            for qy in GAUSS_POINTS:
                for qx in GAUSS_POINTS:
                    shape, gradient = shape_data(qx, qy, qz, hx, hy, hz)
                    current_q = shape @ current_e
                    previous_q = shape @ previous_e
                    grad_pw = current_e[:, 0] @ gradient
                    grad_pc = current_e[:, 1] @ gradient
                    sw, sc, rho_w, rho_c, krw, krc = constitutive(
                        current_q[0], current_q[1]
                    )
                    sw_old, sc_old, rho_w_old, rho_c_old, _, _ = constitutive(
                        previous_q[0], previous_q[1]
                    )
                    accumulation_w = p["porosity"] * (
                        sw * rho_w - sw_old * rho_w_old
                    ) / dt
                    accumulation_c = p["porosity"] * (
                        sc * rho_c - sc_old * rho_c_old
                    ) / dt
                    conductivity_w = (
                        rho_w * krw * p["permeability"] / p["mu_w"]
                    )
                    conductivity_c = (
                        rho_c * krc * p["permeability"] / p["mu_c"]
                    )
                    weight = determinant * GAUSS_WEIGHT**3
                    local[:, 0] += weight * (
                        accumulation_w * shape
                        + conductivity_w * (gradient @ grad_pw)
                    )
                    local[:, 1] += weight * (
                        accumulation_c * shape
                        + conductivity_c * (gradient @ grad_pc)
                    )
        for local_node, node in enumerate(element):
            result[2 * node : 2 * node + 2] += local[local_node]
    return result


def boundary_state(state, coordinates, time, ramp_duration, injection_pressure):
    ramp = min(1.0, max(0.0, time / ramp_duration)) if ramp_duration > 0 else 1.0
    left_pc = 15.1e6 + ramp * (injection_pressure - 15.1e6)
    left = np.isclose(coordinates[:, 0], 0.0)
    right = np.isclose(coordinates[:, 0], 1.0)
    state.reshape(-1, 2)[left, 0] = 15e6
    state.reshape(-1, 2)[left, 1] = left_pc
    state.reshape(-1, 2)[right, 0] = 15e6
    state.reshape(-1, 2)[right, 1] = 15.1e6


def python_solve(dt, end_time, ramp_duration, injection_pressure):
    coordinates, elements = structured_hex8()
    state = np.tile((15e6, 15.1e6), coordinates.shape[0]).astype(np.float64)
    boundary = np.isclose(coordinates[:, 0], 0.0) | np.isclose(
        coordinates[:, 0], 1.0
    )
    free = np.repeat(~boundary, 2)
    time = 0.0
    while time < end_time - 1e-14:
        step_dt = min(dt, end_time - time)
        previous = state.copy()
        time += step_dt
        boundary_state(state, coordinates, time, ramp_duration, injection_pressure)
        for _ in range(20):
            assembled = residual(state, previous, step_dt, coordinates, elements)
            reduced = assembled[free]
            if np.linalg.norm(reduced) <= 1e-9:
                break
            free_indices = np.flatnonzero(free)
            jacobian = np.empty((free_indices.size, free_indices.size))
            for column, dof in enumerate(free_indices):
                perturbation = max(1.0, abs(state[dof]) * 1e-7)
                perturbed = state.copy()
                perturbed[dof] += perturbation
                jacobian[:, column] = (
                    residual(
                        perturbed, previous, step_dt, coordinates, elements
                    )[free]
                    - reduced
                ) / perturbation
            update = np.linalg.solve(jacobian, reduced)
            state[free] -= update
            boundary_state(
                state, coordinates, time, ramp_duration, injection_pressure
            )
        else:
            raise RuntimeError("direct Python Newton solve did not converge")
    return state


def read_state(output):
    restart = os.path.join(output, "restart")
    candidates = (
        os.path.join(restart, "state.float64"),
        os.path.join(restart, "state.real64"),
        os.path.join(restart, "state.double"),
    )
    for path in candidates:
        if os.path.exists(path):
            return np.fromfile(path, dtype=np.float64)
    raise FileNotFoundError("unable to find double restart state in %s" % restart)


def run_generated(driver, output, dt, args):
    shutil.rmtree(output, ignore_errors=True)
    env = os.environ.copy()
    env.update(
        {
            "SFEM_NX": "2",
            "SFEM_NY": "1",
            "SFEM_NZ": "1",
            "SFEM_DT": repr(dt),
            "SFEM_MIN_DT": repr(dt / 1024.0),
            "SFEM_T_END": repr(args.end_time),
            "SFEM_RAMP_DURATION": repr(args.ramp_duration),
            "SFEM_INJECTION_CO2_PRESSURE": repr(args.injection_pressure),
            "SFEM_NL_ATOL": "1e-10",
            "SFEM_NL_RTOL": "1e-10",
            "SFEM_LS_RTOL": "1e-8",
            "SFEM_BENCHMARK_REPEATS": str(args.benchmark_repeats),
        }
    )
    completed = subprocess.run(
        (driver, "GENERATE", output),
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with open(
        os.path.join(output, "performance.csv"), encoding="utf-8"
    ) as performance_file:
        performance = performance_file.read()
    with open(
        os.path.join(output, "mass_balance.csv"), encoding="utf-8"
    ) as balance_file:
        balance = balance_file.read()
    return read_state(output), completed.stdout, performance, balance


def parse_max_balance(text):
    values = [
        abs(float(row["interior_error"]))
        for row in csv.DictReader(io.StringIO(text))
    ]
    return max(values, default=0.0)


def last_balance_lines(text):
    lines = text.strip().splitlines()
    return lines[:1] + lines[-2:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--end-time", type=float, default=0.2)
    parser.add_argument("--ramp-duration", type=float, default=0.2)
    parser.add_argument("--injection-pressure", type=float, default=15.2e6)
    parser.add_argument("--benchmark-repeats", type=int, default=100)
    parser.add_argument("--relative-tolerance", type=float, default=2e-7)
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    generated = []
    performance_reports = []
    balance_reports = []
    for level, factor in enumerate((1.0, 0.5, 0.25)):
        run_dt = args.dt * factor
        state, _, performance, balance = run_generated(
            args.driver,
            os.path.join(args.work_dir, "dt_%d" % level),
            run_dt,
            args,
        )
        generated.append(state)
        performance_reports.append(performance)
        balance_reports.append(balance)
        print(
            "generated_run dt=%.17g output=%s"
            % (run_dt, os.path.join(args.work_dir, "dt_%d" % level))
        )

    reference = python_solve(
        args.dt, args.end_time, args.ramp_duration, args.injection_pressure
    )
    absolute_error = np.linalg.norm(generated[0] - reference)
    reference_norm = max(np.linalg.norm(reference), 1.0)
    relative_error = absolute_error / reference_norm
    coarse_difference = np.linalg.norm(generated[0] - generated[1])
    fine_difference = np.linalg.norm(generated[1] - generated[2])
    temporal_order = (
        math.log(coarse_difference / fine_difference, 2.0)
        if fine_difference > 0.0 and coarse_difference > 0.0
        else float("nan")
    )
    max_balance = max(parse_max_balance(report) for report in balance_reports)

    with open(args.summary, "w", encoding="utf-8") as output:
        output.write("# Generated two-phase flow verification\n\n")
        output.write("| Check | Result |\n|---|---:|\n")
        output.write("| Python/generated relative error | %.6e |\n" % relative_error)
        output.write("| `dt` to `dt/2` difference | %.6e |\n" % coarse_difference)
        output.write("| `dt/2` to `dt/4` difference | %.6e |\n" % fine_difference)
        output.write("| Observed temporal order | %.6f |\n" % temporal_order)
        output.write("| Maximum interior mass-balance residual | %.6e |\n" % max_balance)
        output.write("\n## Final accepted-step balance\n\n```csv\n")
        output.write("\n".join(last_balance_lines(balance_reports[-1])))
        output.write("\n```\n")
        output.write("\n## Generated kernel performance\n\n```csv\n")
        output.write(performance_reports[-1].rstrip())
        output.write("\n```\n")

    print("python_generated_relative_error=%.17g" % relative_error)
    print("temporal_order=%.17g" % temporal_order)
    print("max_mass_balance_error=%.17g" % max_balance)
    print("summary=%s" % args.summary)
    if not np.isfinite(relative_error) or relative_error > args.relative_tolerance:
        raise SystemExit("Python/generated comparison exceeded tolerance")
    if not np.isfinite(temporal_order) or temporal_order < 0.7:
        raise SystemExit("temporal convergence order is below 0.7")


if __name__ == "__main__":
    main()
