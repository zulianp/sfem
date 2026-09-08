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
# Node counts: the study specifies 40x8x8, 80x16x16 and 160x32x32 elements.
RESOLUTIONS = {"coarse": (41, 9, 9), "medium": (81, 17, 17), "fine": (161, 33, 33)}
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


def collect_runs(runs):
    """Combine split jobs on the same generated mesh, never silently pick duplicate cases."""
    case = yaml.safe_load((runs[0] / "case.yaml").read_text())
    results, signature = {}, None
    modes = set()
    for folder in runs:
        saved = yaml.safe_load((folder / "case.yaml").read_text())
        if saved != case:
            raise ValueError(f"{folder}: YAML settings differ; compare identical time/material/loading settings")
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            if len(runs) != 1:
                raise ValueError("Combining jobs requires manifest.json in every run")
            return case, {name: folder / name / case["output"]["path"] for name in POLICIES}, "fp64"
        manifest = json.loads(manifest_path.read_text())
        mode = manifest["history_mode"]
        modes.add(mode)
        # The prefix contains machine-specific Python/script/output paths, not mesh parameters.
        current = (manifest["mesh_command"][4:], manifest["executable_sha256"])
        if signature is not None and current != signature:
            raise ValueError(f"{folder}: mesh or executable differs from the other runs")
        signature = current
        expected = manifest.get("requested_cases", list(manifest["runs"]))
        if set(expected) != set(manifest["runs"]):
            raise ValueError(f"{folder}: job is incomplete; not all requested cases were recorded")
        for name, record in manifest["runs"].items():
            if name not in POLICIES or mode not in ("per_qp", "per_elem"):
                raise ValueError(f"{folder}: unknown history mode or policy")
            if record["returncode"] != 0 or not record.get("complete", True):
                raise ValueError(f"{folder}/{name}: failed or incomplete case")
            if (record["storage"], record["scaling"]) != POLICIES[name]:
                raise ValueError(f"{folder}/{name}: storage metadata does not match the case label")
            key = f"{mode}_{name}"
            if key in results:
                raise ValueError(f"Duplicate case {key}; select non-overlapping jobs")
            results[key] = folder / name / case["output"]["path"]
    reference = "per_qp_fp64" if "per_qp" in modes else "per_elem_fp64"
    if reference not in results or len(results) < 2:
        raise ValueError(f"Comparison needs {reference} and at least one candidate")
    if len(modes) == 1:
        prefix = next(iter(modes)) + "_"
        results = {key.removeprefix(prefix): path for key, path in results.items()}
        reference = "fp64"
    return case, results, reference


def compare_runs(out, runs=None, make_plots=True):
    case, results, reference_name = collect_runs(runs or [out])
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
    if (plots / "comparison.json").exists():
        raise FileExistsError(f"Comparison already complete: {plots}; use --plot-only or a new --out with --runs")
    plots.mkdir(parents=True, exist_ok=True)
    reference = histories[reference_name]
    summaries, scalar_tables, fields = [], [], {}
    for name, data in histories.items():
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
        if name == reference_name:
            continue
        scalar_tables.append(errors.assign(candidate=name))
        fields[name] = compare_displacement(
            results[reference_name], results[name], plots / name, "out", reference_name, name,
            make_plots=False)
    pd.DataFrame(summaries).to_csv(plots / "torsion_summary.csv", index=False)
    pd.concat(scalar_tables, ignore_index=True).to_csv(plots / "torsion_errors.csv", index=False)
    pd.concat([data.assign(candidate=name) for name, data in histories.items()],
              ignore_index=True).to_csv(plots / "torsion_responses.csv", index=False)
    pd.concat([data.assign(candidate=name) for name, data in fields.items()],
              ignore_index=True).to_csv(plots / "coupled_comparison.csv", index=False)
    (plots / "comparison.json").write_text(json.dumps({
        "case": case, "reference": reference_name, "runs": [str(p) for p in (runs or [out])],
    }, indent=2) + "\n")
    if make_plots:
        plot_results(out)
    print(f"[done] Validated comparison tables: {plots}", flush=True)


def plot_results(out):
    """Plot compact comparison CSVs; no mesh, executable or raw fields needed."""
    plots = out / "compare"
    metadata = json.loads((plots / "comparison.json").read_text())
    case, reference_name = metadata["case"], metadata["reference"]
    responses = pd.read_csv(plots / "torsion_responses.csv")
    errors = pd.read_csv(plots / "torsion_errors.csv")
    histories = dict(tuple(responses.groupby("candidate", sort=False)))
    reference = histories[reference_name]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (name, data) in enumerate(histories.items()):
        style = ("-", "--", "-.", ":")[i % 4]
        color = "black" if name == reference_name else f"C{i % 10}"
        axes[0, 0].plot(data.time, data.torque, color=color, linestyle=style, label=name)
        axes[0, 1].plot(data.time, data.uy, color=color, linestyle=style, label=name)
        if name == reference_name:
            continue
        selected = errors[errors.candidate == name]
        for ax, column in zip(axes[1], ("torque_peak_normalized_percent", "control_peak_normalized_percent")):
            values = selected[column].to_numpy()
            ax.plot(selected.time, np.where(values > 0, values, np.nan), color=color, linestyle=style,
                    marker=("o", "s", "^", "x")[i % 4], markersize=3,
                    markevery=max(1, len(data) // 15), label=name)
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
        ax.legend(fontsize=6 if len(histories) > 5 else 8, ncol=2 if len(histories) > 5 else 1)
    integrator = case.get("dynamics", {}).get("type", "quasi_static")
    fig.suptitle(f"Torsion release ({integrator}); reference: {reference_name}")
    fig.text(0.5, 0.01, "Dotted vertical line, if shown: release. Zero log errors omitted; retained in CSV.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for extension in ("png", "pdf"):
        fig.savefig(plots / f"torsion_comparison.{extension}", dpi=180)
    plt.close(fig)
    fields = dict(tuple(pd.read_csv(plots / "coupled_comparison.csv").groupby("candidate", sort=False)))
    plot_comparisons(fields, plots, reference_name)
    print(f"[done] Plots and error tables: {plots}", flush=True)


def run(out, exe, end_time=None, resolution=None, history_mode="per_qp", cases=None, run_only=False):
    cases = list(POLICIES) if cases is None else list(cases)
    if history_mode not in ("per_qp", "per_elem") or not cases or len(set(cases)) != len(cases) or any(
            name not in POLICIES for name in cases):
        raise ValueError("Select per_qp/per_elem and a nonempty list of unique known cases")
    if not run_only and ("fp64" not in cases or len(cases) < 2):
        raise ValueError("Automatic comparison requires fp64 and a candidate; use --run-only for split/debug jobs")
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
    env["SFEM_HISTORY_MODE"] = history_mode
    env.setdefault("SFEM_HISTORY_CHECK", "1")
    nodes = RESOLUTIONS[resolution] if resolution is not None else (
        env.get("PRONY_NX", "16"), env.get("PRONY_NY", "5"), env.get("PRONY_NZ", "5"))
    mesh_command = [sys.executable, str(ROOT / "python/sfem/mesh/box_mesh.py"), str(out / "mesh"),
                    "--cell_type=HEX8", "-x", str(nodes[0]),
                    "-y", str(nodes[1]), "-z", str(nodes[2]),
                    "--width=1.0", "--height=0.2", "--depth=0.2"]
    subprocess.run(mesh_command, check=True, env=env)
    manifest = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "executable": str(exe), "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "case_sha256": hashlib.sha256((out / "case.yaml").read_bytes()).hexdigest(),
        "source_case": str(CASE), "dynamics": case["dynamics"], "time": case["time"],
        "mesh_command": mesh_command, "resolution": resolution,
        "omp_num_threads": env["OMP_NUM_THREADS"],
        "history_mode": history_mode, "history_check": env["SFEM_HISTORY_CHECK"],
        "requested_cases": cases, "runs": {},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in cases:
        storage, scaling = POLICIES[name]
        folder = out / name
        folder.mkdir()
        (folder / "mesh").symlink_to("../mesh", target_is_directory=True)
        shutil.copy2(out / "case.yaml", folder / "case.yaml")
        env.update(SFEM_HISTORY_STORAGE=storage, SFEM_HISTORY_SCALING=scaling)
        print(f"[run] {history_mode}/{name}: {storage}/{scaling}; log: {folder / 'run.log'}", flush=True)
        start = time.monotonic()
        manifest["runs"][name] = {"storage": storage, "scaling": scaling,
                                    "returncode": None, "complete": False}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (folder / "run.log").open("w") as log:
            completed = subprocess.run([str(exe), "case.yaml"], cwd=folder, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        manifest["runs"][name] = {"storage": storage, "scaling": scaling,
                                    "returncode": completed.returncode,
                                    "complete": False,
                                    "wall_seconds": time.monotonic() - start}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError(f"{name} failed ({completed.returncode}); see {folder / 'run.log'}")
        read_history(folder / case["output"]["path"] / case["output"]["history_csv"], case)
        manifest["runs"][name]["complete"] = True
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[ok] {name}: complete and all recorded steps converged", flush=True)
    if not run_only:
        compare_runs(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New run directory; never overwrites simulations")
    parser.add_argument("--exe", type=Path, default=os.environ.get(
        "PRONY_EXE", str(ROOT / "spikes/prony-series/build/prony_visco_torsion")))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plot-only", action="store_true", help="Plot cached comparison CSVs locally (no raw fields)")
    action.add_argument("--compare-only", action="store_true", help="Validate raw outputs and compute compact CSVs, without plots")
    action.add_argument("--run-only", action="store_true", help="Run selected cases without comparison or plots")
    parser.add_argument("--runs", nargs="+", type=Path, help="Input run directories for --compare-only; default: --out")
    parser.add_argument("--history-mode", choices=("per_qp", "per_elem"), default="per_qp")
    parser.add_argument("--cases", nargs="+", choices=POLICIES, help="Selected policies; default: all five")
    parser.add_argument("--end-time", type=float, help="Shorten the run for testing; only the saved YAML copy is changed")
    parser.add_argument("--resolution", choices=RESOLUTIONS,
                        help="Study mesh preset; overrides PRONY_NX/NY/NZ. Omit for the existing custom/smoke mesh")
    args = parser.parse_args()
    if args.runs and not args.compare_only:
        parser.error("--runs requires --compare-only")
    if args.plot_only or args.compare_only:
        if args.end_time is not None or args.resolution is not None or args.cases or args.history_mode != "per_qp":
            parser.error("Postprocessing reads saved settings; do not pass simulation overrides")
        if args.compare_only:
            compare_runs(args.out.resolve(), [p.resolve() for p in args.runs] if args.runs else None, make_plots=False)
        elif (args.out / "compare/comparison.json").exists():
            plot_results(args.out.resolve())
        else:
            compare_runs(args.out.resolve())  # Compatibility with older runs that still have raw fields.
    else:
        run(args.out.resolve(), args.exe.resolve(), args.end_time, args.resolution,
            args.history_mode, args.cases, args.run_only)


if __name__ == "__main__":
    main()
