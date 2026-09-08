#!/usr/bin/env python3
"""Compare five history-storage policies on Newmark torsion with inertia."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from compare_mr_visco_history import (
    compare_displacement, find_displacement_files, load_displacement,
    plot_comparisons, plt, read_times,
)

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "spikes/prony-series/cases/newmark_torsion_release.yaml"
POLICIES = {
    "fp64": ("float64", "none"),
    "fp32": ("float32", "none"),
    "fp16": ("float16", "none"),
    "fp16_tensor": ("float16", "tensor"),
    "fp16_element_prony": ("float16", "element_prony"),
}


def load_case(end_time=None):
    case = yaml.safe_load(CASE.read_text())
    if end_time is not None:
        dt = case["time"]["dt"]
        if not np.isfinite(end_time) or end_time < dt or end_time > case["time"]["t_end"]:
            raise ValueError("--end-time must be finite and between dt and the YAML end time")
        steps = end_time / dt
        if not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
            raise ValueError("--end-time must be an integer multiple of dt")
        case["time"]["t_end"] = end_time
    return case


def read_history(path, case):
    data = pd.read_csv(path)
    columns = ["time", "angle", "released", "torque", "ux", "uy", "uz",
               "newton_it", "lin_it", "gnorm"]
    values = data[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite history values")
    dt, end = case["time"]["dt"], case["time"]["t_end"]
    expected = np.arange(1, int(np.ceil(end / dt)) + 1) * dt
    if len(data) != len(expected) or not np.allclose(data.time, expected, rtol=0, atol=1e-10):
        raise ValueError(f"{path}: incomplete or misaligned time history")
    if np.any(data.gnorm < 0) or np.any(data.gnorm >= case["solver"]["newton"]["tol"]):
        raise ValueError(f"{path}: contains unconverged steps; do not use for precision comparison")
    released = expected >= case["torsion"]["release"]["time"]
    if not np.array_equal(data.released.to_numpy(), released):
        raise ValueError(f"{path}: unexpected release schedule")
    return data


def scalar_errors(reference, candidate):
    if not np.allclose(reference.time, candidate.time, rtol=0, atol=1e-10):
        raise ValueError("Scalar histories are not time-aligned")
    if not np.array_equal(reference.released, candidate.released):
        raise ValueError("Release schedules differ")
    torque_peak = np.abs(reference.torque).max()
    ref_u = reference[["ux", "uy", "uz"]].to_numpy()
    cand_u = candidate[["ux", "uy", "uz"]].to_numpy()
    u_peak = np.linalg.norm(ref_u, axis=1).max()
    if torque_peak <= 0 or u_peak <= 0:
        raise ValueError("Torsion reference must have nonzero peak torque and displacement")
    torque_error = np.abs(candidate.torque.to_numpy() - reference.torque.to_numpy())
    u_error = np.linalg.norm(cand_u - ref_u, axis=1)
    return pd.DataFrame({
        "time": reference.time.to_numpy(),
        "released": reference.released.to_numpy(),
        "torque_abs_error": torque_error,
        "torque_peak_normalized_percent": 100 * torque_error / torque_peak,
        "control_vector_abs_error": u_error,
        "control_peak_normalized_percent": 100 * u_error / u_peak,
    })


def compare_runs(out):
    case = yaml.safe_load((out / "case.yaml").read_text())
    results = {name: out / name / case["output"]["path"] for name in POLICIES}
    histories = {name: read_history(path / case["output"]["history_csv"], case)
                 for name, path in results.items()}
    # The driver exports an initial zero field and always exports the final step.
    # Check exact coverage before reusing the older, looser field-comparison helper.
    for name, path in results.items():
        files = find_displacement_files(path / "out")
        times = read_times(path / "out", len(files[0]))
        if not np.isclose(times[0], 0, rtol=0, atol=1e-10) or not np.isclose(
                times[-1], histories[name].time.iloc[-1], rtol=0, atol=1e-10):
            raise ValueError(f"{path}: displacement output does not cover the entire run")
        if any(np.any(component != 0) for component in load_displacement(files, 0)):
            raise ValueError(f"{path}: expected the initial zero-displacement state")

    plots = out / "compare"
    plots.mkdir(exist_ok=True)
    reference = histories["fp64"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    summaries, scalar_tables, fields = [], [], {}
    for i, (name, data) in enumerate(histories.items()):
        style = ("-", "--", "-.", ":", "--")[i]
        color = ("black", "tab:blue", "tab:orange", "tab:green", "tab:red")[i]
        axes[0, 0].plot(data.time, data.torque, color=color, linestyle=style, label=name)
        axes[0, 1].plot(data.time, data.uy, color=color, linestyle=style, label=name)
        errors = scalar_errors(reference, data)
        summaries.append({
            "candidate": name,
            "peak_torque_abs_error": errors.torque_abs_error.max(),
            "peak_torque_error_percent_of_ref_peak": errors.torque_peak_normalized_percent.max(),
            "peak_control_vector_abs_error": errors.control_vector_abs_error.max(),
            "peak_control_error_percent_of_ref_peak": errors.control_peak_normalized_percent.max(),
            "total_newton_iterations": data.newton_it.sum(),
            "total_linear_iterations": data.lin_it.sum(),
            "max_gnorm": data.gnorm.max(),
        })
        if name == "fp64":
            continue
        scalar_tables.append(errors.assign(candidate=name))
        for ax, column in zip(axes[1], ("torque_peak_normalized_percent", "control_peak_normalized_percent")):
            values = errors[column].to_numpy()
            ax.plot(data.time, np.where(values > 0, values, np.nan), color=color, linestyle=style,
                    marker=("o", "s", "^", "x")[i - 1], markersize=3,
                    markevery=max(1, len(data) // 15), label=name)
        fields[name] = compare_displacement(
            results["fp64"], results[name], plots / name, "out", "fp64", name)
    axes[0, 0].set(title="Reaction torque", ylabel="Torque (model units)")
    axes[0, 1].set(title="Control-point signed displacement", ylabel=r"$u_y$ (length units)")
    axes[1, 0].set(title="Torque error / FP64 peak torque", ylabel="Peak-normalized error (%)")
    axes[1, 1].set(title="Control-vector error / FP64 peak vector norm", ylabel="Peak-normalized error (%)")
    for ax in axes[1]:
        if any(np.any(line.get_ydata() > 0) for line in ax.lines):
            ax.set_yscale("log")
    for ax in axes.flat:
        if case["torsion"]["release"]["time"] <= reference.time.iloc[-1]:
            ax.axvline(case["torsion"]["release"]["time"], color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    integrator = case.get("dynamics", {}).get("type", "quasi_static")
    fig.suptitle(f"Torsion release ({integrator}): coupled history-storage comparison")
    fig.text(0.5, 0.01, "Dotted vertical line, if shown: release. Zero log errors omitted; retained in CSV.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for extension in ("png", "pdf"):
        fig.savefig(plots / f"torsion_comparison.{extension}", dpi=180)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(plots / "torsion_summary.csv", index=False)
    pd.concat(scalar_tables, ignore_index=True).to_csv(plots / "torsion_errors.csv", index=False)
    plot_comparisons(fields, plots, "fp64")
    print(f"[done] Plots and error tables: {plots}", flush=True)


def run(out, exe, end_time=None):
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; use a new --out or --plot-only")
    if not exe.is_file() or not os.access(exe, os.X_OK):
        raise FileNotFoundError(f"Build torsion first; missing executable: {exe}")
    case = load_case(end_time)
    out.mkdir(parents=True)
    if end_time is None:
        shutil.copy2(CASE, out / "case.yaml")
    else:
        (out / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False))
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env["SFEM_HISTORY_MODE"] = "per_qp"
    env.setdefault("SFEM_HISTORY_CHECK", "1")
    mesh_command = [sys.executable, str(ROOT / "python/sfem/mesh/box_mesh.py"), str(out / "mesh"),
                    "--cell_type=HEX8", "-x", env.get("PRONY_NX", "16"),
                    "-y", env.get("PRONY_NY", "5"), "-z", env.get("PRONY_NZ", "5"),
                    "--width=1.0", "--height=0.2", "--depth=0.2"]
    subprocess.run(mesh_command, check=True, env=env)
    manifest = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "executable": str(exe), "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "case_sha256": hashlib.sha256((out / "case.yaml").read_bytes()).hexdigest(),
        "source_case": str(CASE), "dynamics": case["dynamics"], "time": case["time"],
        "mesh_command": mesh_command, "omp_num_threads": env["OMP_NUM_THREADS"],
        "history_mode": "per_qp", "history_check": env["SFEM_HISTORY_CHECK"], "runs": {},
    }
    for name, (storage, scaling) in POLICIES.items():
        folder = out / name
        folder.mkdir()
        (folder / "mesh").symlink_to(out / "mesh", target_is_directory=True)
        shutil.copy2(out / "case.yaml", folder / "case.yaml")
        env.update(SFEM_HISTORY_STORAGE=storage, SFEM_HISTORY_SCALING=scaling)
        print(f"[run] {name}: {storage}/{scaling}; log: {folder / 'run.log'}", flush=True)
        start = time.monotonic()
        with (folder / "run.log").open("w") as log:
            completed = subprocess.run([str(exe), "case.yaml"], cwd=folder, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        manifest["runs"][name] = {"storage": storage, "scaling": scaling,
                                    "returncode": completed.returncode,
                                    "wall_seconds": time.monotonic() - start}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError(f"{name} failed ({completed.returncode}); see {folder / 'run.log'}")
        read_history(folder / case["output"]["path"] / case["output"]["history_csv"], case)
        print(f"[ok] {name}: complete and all recorded steps converged", flush=True)
    compare_runs(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New run directory; never overwrites simulations")
    parser.add_argument("--exe", type=Path, default=os.environ.get(
        "PRONY_EXE", str(ROOT / "spikes/prony-series/build/prony_visco_torsion")))
    parser.add_argument("--plot-only", action="store_true", help="Recreate comparison plots from this runner's outputs")
    parser.add_argument("--end-time", type=float, help="Shorten the run for testing; only the saved YAML copy is changed")
    args = parser.parse_args()
    if args.plot_only:
        if args.end_time is not None:
            parser.error("--end-time cannot be used with --plot-only")
        compare_runs(args.out.resolve())
    else:
        run(args.out.resolve(), args.exe.resolve(), args.end_time)


if __name__ == "__main__":
    main()
