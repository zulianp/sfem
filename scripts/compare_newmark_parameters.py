#!/usr/bin/env python3
"""Local FP64 Newmark screening: fixed 16x5x5 nodes, dt=0.005, T=10."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from run_torsion_history_compare import CASE, ROOT, read_history, plt

PARAMETERS = {"A": (0.25, 0.5), "B": (0.3025, 0.6), "C": (0.64, 0.6)}


def run(out):
    exe = ROOT / "spikes/prony-series/build/prony_visco_torsion"
    if not exe.is_file() or not os.access(exe, os.X_OK):
        raise FileNotFoundError(exe)
    out.mkdir(parents=True, exist_ok=False)
    case = yaml.safe_load(CASE.read_text())
    case["time"] = {"dt": 0.005, "t_end": 10.0}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               SFEM_HISTORY_MODE="per_qp", SFEM_HISTORY_STORAGE="float64",
               SFEM_HISTORY_SCALING="none", SFEM_HISTORY_CHECK="1")
    subprocess.run([sys.executable, str(ROOT / "python/sfem/mesh/box_mesh.py"),
                    str(out / "mesh"), "--cell_type=HEX8", "-x", "16", "-y", "5", "-z", "5",
                    "--width=1", "--height=0.2", "--depth=0.2"], env=env, check=True)
    manifest = {"git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "executable": str(exe), "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "source_yaml_sha256": hashlib.sha256(CASE.read_bytes()).hexdigest(),
        "nodes": [16, 5, 5], "elements": 240, "history_storage": "float64",
        "history_check": "1", "omp_num_threads": 1, "runs": {}}
    for name, (beta, gamma) in PARAMETERS.items():
        folder = out / name
        folder.mkdir()
        (folder / "mesh").symlink_to(out / "mesh", target_is_directory=True)
        configured = copy.deepcopy(case)
        configured["dynamics"].update(beta=beta, gamma=gamma)
        (folder / "case.yaml").write_text(yaml.safe_dump(configured, sort_keys=False))
        print(f"[run] {name}: beta={beta}, gamma={gamma}", flush=True)
        start = time.monotonic()
        with (folder / "run.log").open("w") as log:
            result = subprocess.run([str(exe), "case.yaml"], cwd=folder, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        status = {"beta": beta, "gamma": gamma, "returncode": result.returncode,
                  "wall_seconds": time.monotonic() - start, "complete": False}
        try:
            read_history(folder / configured["output"]["path"] / configured["output"]["history_csv"], configured)
            status["complete"] = result.returncode == 0
        except (ValueError, FileNotFoundError) as error:
            status["validation_error"] = str(error)
        manifest["runs"][name] = status
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[done] {name}: {status}", flush=True)


def compare(out):
    manifest = json.loads((out / "manifest.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    summaries = []
    for name, (beta, gamma) in PARAMETERS.items():
        case = yaml.safe_load((out / name / "case.yaml").read_text())
        path = out / name / case["output"]["path"] / case["output"]["history_csv"]
        if not path.exists():
            continue
        data = pd.read_csv(path)
        if data.empty:
            continue
        assert np.isfinite(data.to_numpy(dtype=float)).all(), path
        assert np.all(np.diff(data.time) > 0), path
        status = manifest["runs"][name]
        label = f"{name}: beta={beta}, gamma={gamma}"
        if not status["complete"]:
            label += f" (incomplete, t={data.time.iloc[-1]:g})"
        style = {"A": "-", "B": "--", "C": ":"}[name]
        axes[0, 0].plot(data.time, data.torque, style, label=label)
        early = data[data.time <= 0.2]
        axes[0, 1].plot(early.time, early.torque, style, label=label)
        axes[1, 0].plot(data.time, data.uy, style, label=label)
        released = data[data.time >= case["torsion"]["release"]["time"]]
        axes[1, 1].plot(released.time, released.uy, style, label=label)
        summaries.append(dict(candidate=name, **status, last_time=data.time.iloc[-1],
                              recorded_steps=len(data), max_gnorm=data.gnorm.max(),
                              unconverged_steps=int((data.gnorm >= case["solver"]["newton"]["tol"]).sum()),
                              total_newton_iterations=int(data.newton_it.sum()),
                              total_linear_iterations=int(data.lin_it.sum()),
                              peak_abs_torque=data.torque.abs().max(), final_uy=data.uy.iloc[-1]))
    for ax, title, ylabel in zip(axes.flat,
            ["Reaction torque: all recorded steps", "Startup torque: first 0.2 s",
             "Control-point signed displacement", "Control-point displacement after release"],
            ["Torque [model units]", "Torque [model units]", "uy [length units]", "uy [length units]"]):
        ax.set(title=title, xlabel="Time [s]", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("FP64 Newmark parameter screening: 240 HEX8 elements, dt=0.005 s")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / f"newmark_parameters.{extension}", dpi=180)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if not args.plot_only:
        run(out)
    compare(out)
