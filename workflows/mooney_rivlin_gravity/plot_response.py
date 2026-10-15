#!/usr/bin/env python3
"""Response and Newton history of a sfem_MooneyRivlinGravityTest run.

usage: plot_response.py cantilever|contact <output-dir> <fields-dir>

Reads <output-dir>/run.log and the raw fields in <fields-dir>; writes response.csv,
newton.csv and gravity_response.png to <output-dir>.

  cantilever  the mean displacement of the free end
  contact     the mean descent of the block's underside, and the obstacle's total reaction
              over the block's weight, which averages to 1 once the block is on the obstacle
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

DTYPES = {"float64": np.float64, "float32": np.float32, "raw": np.float64}

# The tetrahedra of a HEX8 around its 0-6 diagonal.
HEX8_TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def read_newton(log: Path):
    steps = []
    for line in log.read_text().splitlines():
        # Numbers only: stderr warnings can land in the middle of a line.
        m = re.match(r"Step (\d+): t=([0-9.eE+-]+)", line)
        if m:
            steps.append({"step": int(m.group(1)), "t": float(m.group(2)), "iters": 0, "kkt": float("nan")})
            continue
        m = re.match(r"\s+Iter (\d+): \|R\|=\S+ \|KKT\|=([0-9.eE+-]+|nan)", line)
        if m and steps:
            steps[-1]["iters"] = int(m.group(1))
            steps[-1]["kkt"] = float(m.group(2))
    return steps


def read_series(fields: Path, name: str, component: int):
    files = []
    for path in fields.glob(f"{name}.{component}.*.*"):
        _, _, counter, suffix = path.name.split(".")
        files.append((int(counter), path, DTYPES[suffix]))
    files.sort()
    return [np.fromfile(path, dtype=dtype) for _, path, dtype in files]


def hex8_volume(fields: Path, points):
    idx = np.stack([np.fromfile(fields / f"i{k}.int32", dtype=np.int32) for k in range(8)])
    p = np.stack(points)  # 3 x nodes
    volume = 0.0
    for a, b, c, d in HEX8_TETS:
        e1, e2, e3 = (p[:, idx[k]] - p[:, idx[a]] for k in (b, c, d))
        volume += np.abs(np.einsum("ij,ij->j", e1, np.cross(e2, e3, axis=0))).sum() / 6
    return volume


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["cantilever", "contact"])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("fields_dir", type=Path)
    args = parser.parse_args()

    out, fields = args.output_dir, args.fields_dir
    log = (out / "run.log").read_text()

    mesh = re.search(r"Mesh: (\d+) nodes, (\d+) DOFs", log)
    n_elements = re.search(r"n_elements: (\d+)", (fields / "meta.yaml").read_text())
    steps = read_newton(out / "run.log")
    times = np.loadtxt(fields / "time.txt", ndmin=1)
    points = [np.fromfile(fields / f"{c}.float32", dtype=np.float32).astype(np.float64) for c in "xyz"]
    disp = [read_series(fields, "disp", c) for c in range(3)]
    n_frames = min(len(times), *(len(d) for d in disp))
    times = times[:n_frames]

    failures = []
    if not all(np.isfinite(d).all() for series in disp for d in series):
        failures.append("non-finite displacement")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = f"{n_elements.group(1)} HEX8, {mesh.group(1)} nodes, {mesh.group(2)} dofs"
    if args.case == "cantilever":
        x = points[0]
        tip = np.abs(x - x.max()) <= 1e-6 * max(1.0, abs(x.max()))
        response = np.array([[disp[c][k][tip].mean() for c in range(3)] for k in range(n_frames)])
        header = "t,ux,uy,uz"
        fig, (ax_r, ax_n) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1])
        ax_r.plot(times, response[:, 1], lw=2)
        ax_r.set_ylabel("tip deflection $u_y$")
        ax_r.set_title(f"Mooney-Rivlin Prony cantilever under gravity: {size}")
        axes = [ax_r]
        summary = f"final tip u_y = {response[-1, 1]:.6g}"
    else:
        d = int(os.environ.get("SFEM_CONTACT_DIR", 2))
        density = float(os.environ.get("SFEM_DENSITY", 1000))
        gravity = float(os.environ.get("SFEM_GRAVITY", 9.81))
        weight = density * gravity * hex8_volume(fields, points)
        under = np.abs(points[d] - points[d].min()) <= 1e-6 * max(1.0, np.ptp(points[d]))
        force = read_series(fields, "contact_force", d)
        descent = np.array([disp[d][k][under].mean() for k in range(n_frames)])
        reaction = np.array([force[k].sum() / weight for k in range(n_frames)])
        response = np.stack([descent, reaction], axis=1)
        header = "t,underside_displacement,reaction_over_weight"
        fig, (ax_d, ax_f, ax_n) = plt.subplots(3, 1, figsize=(8, 8), sharex=True, height_ratios=[2, 2, 1])
        ax_d.plot(times, descent, lw=2)
        ax_d.set_ylabel(f"underside $u_{'xyz'[d]}$ (mean)")
        ax_d.set_title(f"Mooney-Rivlin Prony block dropped on an obstacle: {size}")
        ax_f.plot(times, reaction, lw=2)
        ax_f.axhline(1, color="0.5", lw=1, ls="--")
        ax_f.set_ylabel("reaction / weight")
        axes = [ax_d, ax_f]
        late = times > times[-1] / 2
        summary = (f"final underside u = {descent[-1]:.6g}, "
                   f"reaction / weight over the second half = {reaction[late].mean():.4f}")

    np.savetxt(out / "response.csv", np.column_stack([times, response]), delimiter=",", header=header, comments="", fmt="%.10g")
    with open(out / "newton.csv", "w") as f:
        f.write("step,t,iterations,final_kkt\n")
        for s in steps:
            f.write(f"{s['step']},{s['t']:.10g},{s['iters']},{s['kkt']:.6g}\n")

    ax_n.step([s["t"] for s in steps], [s["iters"] for s in steps], where="post", lw=1.5)
    ax_n.set_ylabel("Newton iters")
    ax_n.set_xlabel("time")
    for ax in axes + [ax_n]:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "gravity_response.png", dpi=150)

    print(f"{len(steps)} steps, {n_frames} frames, {summary}, max Newton iterations = {max(s['iters'] for s in steps)}")
    print(f"Wrote {out / 'gravity_response.png'}")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
