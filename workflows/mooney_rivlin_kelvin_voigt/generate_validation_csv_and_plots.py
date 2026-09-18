#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sfem_mrkv_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sfem_mrkv_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REFERENCE_HOMOGENEOUS = {
    "HEX8": {
        "interior_nodes": 1,
        "max_interior": 1.453989e-16,
        "max_all": 8.096176e-1,
        "tol": 8.096176e-10,
    },
    "HEX27": {
        "interior_nodes": 27,
        "max_interior": 9.757752e-16,
        "max_all": 3.598301e-1,
        "tol": 3.598301e-10,
    },
}

REFERENCE_OSCILLATOR = {
    "HEX8": {
        "m": 1.0,
        "k": 5.91331182e1,
        "c": 9.85551970e-2,
        "rel_m": 6.920e-12,
        "rel_k": 1.427e-3,
        "rel_c": 1.427e-3,
        "omega0": 7.68980612,
        "delta": 4.92775985e-2,
        "max_q_error": 3.82750275e-6,
    },
    "HEX27": {
        "m": 1.0,
        "k": 5.92166518e1,
        "c": 9.86944197e-2,
        "rel_m": 0.0,
        "rel_k": 1.646e-5,
        "rel_c": 1.646e-5,
        "omega0": 7.69523566,
        "delta": 4.93472098e-2,
        "max_q_error": 3.82750067e-6,
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_test(executable: Path) -> str:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMPI_MCA_btl", "self")
    proc = subprocess.run(
        [str(executable)],
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=True,
    )
    return proc.stdout


def parse_homogeneous(output: str):
    pattern = re.compile(
        r"homogeneous deformation (\w+): interior_nodes=(\d+) "
        r"max_interior=([0-9.eE+-]+) max_all=([0-9.eE+-]+) tol=([0-9.eE+-]+)"
    )
    values = {}
    for match in pattern.finditer(output):
        values[match.group(1)] = {
            "interior_nodes": int(match.group(2)),
            "max_interior": float(match.group(3)),
            "max_all": float(match.group(4)),
            "tol": float(match.group(5)),
        }
    return values


def parse_oscillator(output: str):
    coeff_pattern = re.compile(
        r"linearized oscillator (\w+): m=([0-9.eE+-]+) k=([0-9.eE+-]+) c=([0-9.eE+-]+) "
        r"rel\(m,k,c\)=\(([0-9.eE+-]+), ([0-9.eE+-]+), ([0-9.eE+-]+)\)"
    )
    time_pattern = re.compile(
        r"linearized oscillator (\w+): omega0=([0-9.eE+-]+) delta=([0-9.eE+-]+) "
        r"max_q_error=([0-9.eE+-]+)"
    )
    values = {}
    for match in coeff_pattern.finditer(output):
        values.setdefault(match.group(1), {}).update(
            {
                "m": float(match.group(2)),
                "k": float(match.group(3)),
                "c": float(match.group(4)),
                "rel_m": float(match.group(5)),
                "rel_k": float(match.group(6)),
                "rel_c": float(match.group(7)),
            }
        )
    for match in time_pattern.finditer(output):
        values.setdefault(match.group(1), {}).update(
            {
                "omega0": float(match.group(2)),
                "delta": float(match.group(3)),
                "max_q_error": float(match.group(4)),
            }
        )
    return values


def collect_validation_data(build_dir: Path, skip_tests: bool):
    if skip_tests:
        return dict(REFERENCE_HOMOGENEOUS), dict(REFERENCE_OSCILLATOR)

    homogeneous_exe = build_dir / "sfem_MRKVHomogeneousDeformationValidation"
    oscillator_exe = build_dir / "sfem_MRKVLinearizedOscillationValidation"
    if not homogeneous_exe.exists() or not oscillator_exe.exists():
        missing = [str(p) for p in (homogeneous_exe, oscillator_exe) if not p.exists()]
        raise FileNotFoundError(
            "Missing validation executable(s): "
            + ", ".join(missing)
            + ". Build the validation targets first or pass --skip-tests."
        )

    homogeneous = parse_homogeneous(run_test(homogeneous_exe))
    oscillator = parse_oscillator(run_test(oscillator_exe))
    for element in ("HEX8", "HEX27"):
        if element not in homogeneous or element not in oscillator:
            raise RuntimeError(f"Validation output did not contain complete {element} data")
    return homogeneous, oscillator


def exact_underdamped_q(q0: float, v0: float, omega0: float, delta: float, t: float) -> float:
    omega_d = math.sqrt(omega0 * omega0 - delta * delta)
    return math.exp(-delta * t) * (
        q0 * math.cos(omega_d * t) + ((v0 + delta * q0) / omega_d) * math.sin(omega_d * t)
    )


def oscillator_samples(metrics: dict, steps: int):
    beta = 0.25
    gamma = 0.5
    m = metrics["m"]
    k = metrics["k"]
    c = metrics["c"]
    omega0 = math.sqrt(k / m)
    delta = c / (2.0 * m)
    period = 2.0 * math.pi / omega0
    dt = period / steps
    alpha_a = 1.0 / (beta * dt * dt)
    alpha_v = gamma / (beta * dt)

    q = 1.0
    v = 0.0
    a = -(c * v + k * q) / m
    rows = []
    for step in range(steps + 1):
        t = step * dt
        q_exact = exact_underdamped_q(1.0, 0.0, omega0, delta, t)
        rows.append(
            {
                "step": step,
                "time": t,
                "t_over_T": t / period,
                "q_newmark": q,
                "q_exact": q_exact,
                "error": q - q_exact,
            }
        )

        if step == steps:
            break

        q_hat = q + dt * v + dt * dt * (0.5 - beta) * a
        z = v + dt * (1.0 - gamma) * a - alpha_v * q_hat
        q_new = (m * alpha_a * q_hat - c * z) / (k + c * alpha_v + m * alpha_a)
        v_new = alpha_v * q_new + z
        a_new = alpha_a * (q_new - q_hat)
        q = q_new
        v = v_new
        a = a_new

    return rows


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(output_dir: Path, homogeneous: dict, oscillator: dict, sample_count: int):
    homogeneous_rows = []
    for element, values in homogeneous.items():
        homogeneous_rows.append(
            {
                "element": element,
                "interior_nodes": values["interior_nodes"],
                "max_interior_residual": values["max_interior"],
                "tolerance": values["tol"],
                "max_boundary_reaction": values["max_all"],
            }
        )

    oscillator_rows = []
    sample_rows = []
    for element, values in oscillator.items():
        oscillator_rows.append(
            {
                "element": element,
                "modal_mass": values["m"],
                "modal_stiffness": values["k"],
                "modal_damping": values["c"],
                "relative_mass_error": values["rel_m"],
                "relative_stiffness_error": values["rel_k"],
                "relative_damping_error": values["rel_c"],
                "omega0": values["omega0"],
                "delta": values["delta"],
                "max_q_error": values["max_q_error"],
            }
        )
        for row in oscillator_samples(values, sample_count):
            row_with_element = {"element": element}
            row_with_element.update(row)
            sample_rows.append(row_with_element)

    write_csv(
        output_dir / "homogeneous_residuals.csv",
        ["element", "interior_nodes", "max_interior_residual", "tolerance", "max_boundary_reaction"],
        homogeneous_rows,
    )
    write_csv(
        output_dir / "oscillator_metrics.csv",
        [
            "element",
            "modal_mass",
            "modal_stiffness",
            "modal_damping",
            "relative_mass_error",
            "relative_stiffness_error",
            "relative_damping_error",
            "omega0",
            "delta",
            "max_q_error",
        ],
        oscillator_rows,
    )
    write_csv(
        output_dir / "oscillator_samples.csv",
        ["element", "step", "time", "t_over_T", "q_newmark", "q_exact", "error"],
        sample_rows,
    )
    return oscillator_rows, sample_rows


def save_figure(fig, output_dir: Path, stem: str):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_homogeneous(output_dir: Path, homogeneous: dict):
    elements = list(homogeneous.keys())
    x = np.arange(len(elements))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width, [homogeneous[e]["max_interior"] for e in elements], width, label="max interior residual")
    ax.bar(x, [homogeneous[e]["tol"] for e in elements], width, label="test tolerance")
    ax.bar(x + width, [homogeneous[e]["max_all"] for e in elements], width, label="max boundary reaction")
    ax.set_yscale("log")
    ax.set_xticks(x, elements)
    ax.set_ylabel("Residual norm")
    ax.set_title("Homogeneous finite-strain patch validation")
    ax.grid(True, axis="y", which="both", alpha=0.28)
    ax.legend(frameon=False, ncols=1)
    save_figure(fig, output_dir, "homogeneous_residuals")


def plot_oscillator(output_dir: Path, sample_rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for element in ("HEX8", "HEX27"):
        rows = [r for r in sample_rows if r["element"] == element]
        ax.plot([r["t_over_T"] for r in rows], [r["q_newmark"] for r in rows], label=f"{element} Newmark", linewidth=2.0)
    rows = [r for r in sample_rows if r["element"] == "HEX27"]
    ax.plot([r["t_over_T"] for r in rows], [r["q_exact"] for r in rows], "k--", label="analytical", linewidth=1.6)
    ax.set_xlabel(r"Normalized time, $t/T$")
    ax.set_ylabel(r"Modal displacement, $q$")
    ax.set_title("Linearized damped shear oscillator")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "oscillator_response")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for element in ("HEX8", "HEX27"):
        rows = [r for r in sample_rows if r["element"] == element]
        ax.plot([r["t_over_T"] for r in rows], [r["error"] for r in rows], label=f"{element} error", linewidth=2.0)
    ax.axhline(2e-5, color="0.25", linestyle="--", linewidth=1.2, label="validation threshold")
    ax.axhline(-2e-5, color="0.25", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"Normalized time, $t/T$")
    ax.set_ylabel(r"$q_{\mathrm{Newmark}} - q_{\mathrm{exact}}$")
    ax.set_title("Oscillator pointwise error")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "oscillator_error")


def main():
    parser = argparse.ArgumentParser(description="Generate MRKV Newmark validation CSV tables and plots.")
    parser.add_argument("--build-dir", default=str(repo_root() / "build64"), help="SFEM build directory containing validation executables.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "validation"),
        help="Directory for generated CSV, SVG, and PNG files.",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Use embedded reference metrics instead of running validation executables.")
    parser.add_argument("--samples", type=int, default=2000, help="Newmark samples per oscillator period.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    homogeneous, oscillator = collect_validation_data(Path(args.build_dir).resolve(), args.skip_tests)
    _, sample_rows = write_tables(output_dir, homogeneous, oscillator, args.samples)
    plot_homogeneous(output_dir, homogeneous)
    plot_oscillator(output_dir, sample_rows)

    print(f"Wrote validation CSV and plots to {output_dir}")


if __name__ == "__main__":
    main()
