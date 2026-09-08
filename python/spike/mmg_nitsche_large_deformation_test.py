#!/usr/bin/env python3
"""Large-deformation stress test for Neo-Hookean Nitsche multigrid.

The default case indents the block by three quarters of its height on a
four-level nested mesh.  The solve applies the target through adaptive load
increments guarded by element-Jacobian checks on both coarse corrections and
fine nonlinear smoothing updates.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback

import numpy as np

_SPIKE = os.path.dirname(os.path.abspath(__file__))
if _SPIKE not in sys.path:
    sys.path.insert(0, _SPIKE)

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("OMPI_MCA_btl", "self")

import mmg_nitsche as mg
import mmg_nitsche_neohookean as neo
import nitsche_contact as nc


def configured_solver_args(opts, indent, load_steps):
    args = neo.parse_args([])
    args.nx = int(opts.nx)
    args.ny = int(opts.ny)
    args.levels = int(opts.levels)
    args.indent = float(indent)
    args.load_steps = int(load_steps)
    args.load_reduction = float(opts.load_reduction)
    args.load_min_fraction = float(opts.load_min_fraction)
    args.load_max_accepted = int(opts.load_max_accepted)
    args.load_max_attempts = int(opts.load_max_attempts)
    args.max_iter = int(opts.max_iter)
    args.max_inner_it = int(opts.max_inner_it)
    args.nlsmooth_steps = int(opts.nlsmooth_steps)
    args.mg_pre = int(opts.mg_pre)
    args.mg_post = int(opts.mg_post)
    args.mg_omega = float(opts.mg_omega)
    args.rtol = float(opts.rtol)
    args.atol = float(opts.atol)
    args.gamma0 = float(opts.gamma0)
    args.coarse_tangent = "rediscretized"
    args.coarse_displacement_projection = "injection"
    args.material_linearization = "every-vcycle"
    args.coarse_linesearch = True
    args.coarse_linesearch_mode = "inversion"
    args.coarse_linesearch_reduction = float(opts.coarse_linesearch_reduction)
    args.coarse_linesearch_min_j = float(opts.min_j)
    args.smooth_linesearch = True
    args.smooth_linesearch_reduction = float(opts.smooth_linesearch_reduction)
    args.smooth_linesearch_min_alpha = float(opts.smooth_linesearch_min_alpha)
    args.smooth_linesearch_min_j = float(opts.min_j)
    args.material_tangent = "analytic"
    args.contact_penalty_scaling = str(opts.contact_penalty_scaling)
    args.plot = False
    return args


def summarize_result(label, status, result=None, error=None):
    out = {"label": label, "status": status}
    if result is not None:
        r_hist = np.asarray(result.get("r_hist", []), dtype=np.float64)
        load_history = result.get("load_history", [])
        rejected = [s for s in load_history if not s.get("accepted", False)]
        accepted = [s for s in load_history if s.get("accepted", False)]
        out.update(
            {
                "indent": float(result.get("indent", np.nan)),
                "target_indent": float(result.get("target_indent", result.get("indent", np.nan))),
                "reached_target": bool(result.get("reached_target", True)),
                "final_residual": float(r_hist[-1]) if r_hist.size else None,
                "residual_records": int(r_hist.size),
                "min_J": float(result.get("min_J", np.nan)),
                "penetration_norm": float(result.get("penetration", np.nan)),
                "active_rows": int(result.get("n_active", 0)),
                "load_accepted": len(accepted),
                "load_rejected": len(rejected),
                "force": float(result.get("F", np.nan)),
                "gamma0": float(result.get("gamma0", np.nan)),
                "contact_penalty_scaling": str(result.get("contact_penalty_scaling", "")),
            }
        )
    if error is not None:
        out["error"] = str(error)
        out["traceback"] = traceback.format_exc()
    return out


def run_case(ps, label, args):
    try:
        result = neo.solve_incremental_load(ps, args)
        return result, summarize_result(label, "ok", result=result)
    except Exception as exc:
        return None, summarize_result(label, "failed", error=exc)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-indent", type=float, default=0.30)
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=16)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--load-steps", type=int, default=8)
    p.add_argument("--load-reduction", type=float, default=0.5)
    p.add_argument("--load-min-fraction", type=float, default=1e-3)
    p.add_argument("--load-max-accepted", type=int, default=96)
    p.add_argument("--load-max-attempts", type=int, default=384)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--max-inner-it", type=int, default=6)
    p.add_argument("--nlsmooth-steps", type=int, default=4)
    p.add_argument("--mg-pre", type=int, default=8)
    p.add_argument("--mg-post", type=int, default=8)
    p.add_argument("--mg-omega", type=float, default=0.35)
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--gamma0", type=float, default=50.0)
    p.add_argument("--min-j", type=float, default=1e-6)
    p.add_argument(
        "--contact-penalty-scaling",
        choices=("shear", "normal-tangent"),
        default="shear",
    )
    p.add_argument("--coarse-linesearch-reduction", type=float, default=0.7)
    p.add_argument("--smooth-linesearch-reduction", type=float, default=0.7)
    p.add_argument("--smooth-linesearch-min-alpha", type=float, default=1e-4)
    p.add_argument(
        "--output-dir",
        default="/private/tmp/mmg_large_deformation",
        help="Directory for the JSON summary and final plot.",
    )
    return p.parse_args(argv)


def main(argv=None):
    opts = parse_args(argv)
    os.makedirs(opts.output_dir, exist_ok=True)

    mg.build_level = neo.build_level
    mg.pack_result = neo.pack_result

    ps = nc._load_pysfem()
    ps.init()
    summaries = []
    continued_result = None
    try:
        continued_args = configured_solver_args(opts, opts.target_indent, opts.load_steps)
        continued_result, continued_summary = run_case(ps, "adaptive_continuation", continued_args)
        summaries.append(continued_summary)

        if continued_result is not None:
            plot_path = os.path.join(opts.output_dir, "large_deformation.png")
            neo.plot_result(continued_result, plot_path)
            summaries[-1]["plot"] = plot_path
    finally:
        ps.finalize()

    summary_path = os.path.join(opts.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
        f.write("\n")

    print(json.dumps(summaries, indent=2))
    print(f"summary: {summary_path}")
    return 0 if continued_result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
