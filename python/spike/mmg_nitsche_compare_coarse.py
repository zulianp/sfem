#!/usr/bin/env python3
"""Compare Nitsche contact convergence with and without coarse correction.

The x-axis is accumulated smoothing sweeps.  For the multigrid run this counts
the fine nonlinear smoothing plus the recursive coarse-level smoothing used by
the Galerkin correction.  For the smoother-only run it counts only the fine
nonlinear smoothing because the coarse correction is disabled.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

_SPIKE = os.path.dirname(os.path.abspath(__file__))
if _SPIKE not in sys.path:
    sys.path.insert(0, _SPIKE)

import mmg_nitsche as mg
import mmg_nitsche_neohookean as neo
import nitsche_contact as nc


def coarse_smoothing_sweeps_per_cycle(args, nlevels):
    if args.skip_coarse or nlevels < 3:
        return 0
    sweep_factor = 2 if args.smoother == "sgs" else 1
    sweeps = 0
    visits = 1
    for _ell in range(1, nlevels - 1):
        sweeps += visits * (args.mg_pre + args.mg_post) * sweep_factor
        visits *= args.cycle_type
    return sweeps


def smoothing_axis(result, args, nlevels):
    sweep_factor = 2 if args.smoother == "sgs" else 1
    fine_sweeps = args.nlsmooth_steps * (args.mg_pre + args.mg_post) * sweep_factor
    per_cycle = fine_sweeps + coarse_smoothing_sweeps_per_cycle(args, nlevels)
    cycles = np.cumsum(np.asarray(result["vcycle_hist"], dtype=np.int64))
    return cycles * per_cycle


def run_case(ps, base_args, label, skip_coarse):
    args = copy.copy(base_args)
    args.skip_coarse = bool(skip_coarse)
    args.plot = False
    print(f"case={label} skip_coarse={args.skip_coarse}")
    result = mg.solve_mmg(ps, args)
    nlevels = len(mg.hierarchy_sizes(args.nx, args.ny, args.levels))
    steps = smoothing_axis(result, args, nlevels)
    residuals = np.asarray(result["r_hist"], dtype=np.float64)
    active = np.asarray(result["n_active_hist"], dtype=np.int64)
    return {
        "label": label,
        "args": args,
        "result": result,
        "steps": steps,
        "residuals": residuals,
        "active": active,
        "nlevels": nlevels,
    }


def write_csv(path, cases):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("case", "record", "smoothing_sweeps", "residual", "active_rows"))
        for case in cases:
            for i, (steps, residual, active) in enumerate(
                zip(case["steps"], case["residuals"], case["active"])
            ):
                writer.writerow((case["label"], i + 1, int(steps), residual, int(active)))


def plot_cases(path, cases, args):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    ax_res, ax_active = axs
    colors = ("C0", "C3")
    for color, case in zip(colors, cases):
        label = case["label"]
        steps = case["steps"]
        residuals = case["residuals"]
        active = case["active"]
        ax_res.semilogy(steps, np.maximum(residuals, 1e-30), "o-", color=color, lw=1.8, ms=4, label=label)
        ax_active.plot(steps, active, "s-", color=color, lw=1.5, ms=4, label=label)
    ax_res.axhline(args.atol, color="0.35", ls="--", lw=0.9, label=fr"atol={args.atol:g}")
    ax_res.set_xlabel("accumulated smoothing sweeps")
    ax_res.set_ylabel(r"$\|r\|$")
    ax_res.set_title("residual reduction")
    ax_res.grid(True, which="both", ls=":", alpha=0.35)
    ax_res.legend(loc="best", fontsize=8)
    ax_active.set_xlabel("accumulated smoothing sweeps")
    ax_active.set_ylabel("active rows")
    ax_active.set_title("active-set size")
    ax_active.grid(True, ls=":", alpha=0.35)
    ax_active.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Coarse correction comparison: "
        f"nx={args.nx}, ny={args.ny}, levels={args.levels}, smoother={args.smoother}"
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ny", type=int, default=8)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--r-inner", type=float, default=0.4)
    p.add_argument("--width", type=float, default=1.6)
    p.add_argument("--height", type=float, default=0.4)
    p.add_argument("--gap", type=float, default=0.0)
    p.add_argument("--indent", type=float, default=0.02)
    p.add_argument("--E-block", type=float, default=1.0)
    p.add_argument("--E-obstacle", type=float, default=1.0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--gamma0", type=float, default=50.0)
    p.add_argument("--fd-eps", type=float, default=1e-7)
    p.add_argument(
        "--material-linearization",
        choices=("every-call", "every-vcycle"),
        default="every-call",
        help="Neo-Hookean elastic tangent refresh policy.",
    )
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--max-inner-it", type=int, default=3)
    p.add_argument("--nlsmooth-steps", type=int, default=3)
    p.add_argument("--cycle-type", type=int, default=1, choices=(1, 2))
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--ptol", type=float, default=float("inf"))
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--mg-pre", type=int, default=8)
    p.add_argument("--mg-post", type=int, default=8)
    p.add_argument("--mg-omega", type=float, default=0.25)
    p.add_argument("--smoother", choices=("jacobi", "sgs", "gs", "block", "scalar"), default="jacobi")
    p.add_argument(
        "--coarse-tangent",
        choices=("galerkin", "rediscretized"),
        default="galerkin",
        help="Use inherited Galerkin coarse operators or assemble each coarse tangent at projected displacement.",
    )
    p.add_argument(
        "--coarse-displacement-projection",
        choices=("injection", "l2"),
        default="injection",
        help="Projection used for rediscretized coarse tangents.",
    )
    p.add_argument("--coarse-linesearch", action="store_true")
    p.add_argument(
        "--coarse-linesearch-mode",
        choices=("residual", "inversion"),
        default="residual",
        help="Backtrack coarse corrections by residual decrease or only by positive element Jacobian.",
    )
    p.add_argument("--coarse-linesearch-reduction", type=float, default=0.5)
    p.add_argument("--coarse-linesearch-min-alpha", type=float, default=1e-3)
    p.add_argument("--coarse-linesearch-c1", type=float, default=0.0)
    p.add_argument("--coarse-linesearch-min-j", type=float, default=1e-10)
    p.add_argument("--stagnation-threshold", type=float, default=0.999)
    p.add_argument("--mu-f", type=float, default=0.3)
    p.add_argument("--rigid-block", action="store_true")
    p.add_argument("--rigid-obstacle", action="store_true")
    p.add_argument("--rigid-stiffness", type=float, default=1e4)
    p.add_argument("--biased", action="store_true")
    p.add_argument("--unbiased", action="store_true")
    p.add_argument("--penalty", action="store_true")
    p.add_argument("--lagrange", action="store_true")
    p.add_argument(
        "--plot-output",
        default=os.path.join(_SPIKE, "mmg_nitsche_compare_coarse.png"),
        help="PNG path for the comparison plot.",
    )
    p.add_argument(
        "--csv-output",
        default=os.path.join(_SPIKE, "mmg_nitsche_compare_coarse.csv"),
        help="CSV path for the plotted convergence data.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    mg.build_level = neo.build_level
    mg.pack_result = neo.pack_result
    ps = nc._load_pysfem()
    ps.init()
    try:
        cases = [
            run_case(ps, args, "with coarse correction", False),
            run_case(ps, args, "smoother only", True),
        ]
        write_csv(args.csv_output, cases)
        plot_cases(args.plot_output, cases, args)
        for case in cases:
            print(
                f"{case['label']}: final ||r||={case['residuals'][-1]:.6e} "
                f"smoothing_sweeps={int(case['steps'][-1])} "
                f"active={int(case['active'][-1])}"
            )
    finally:
        ps.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
