#!/usr/bin/env python3
"""Run generated-Stokes FVCA8 verification levels and collect convergence data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build64/generated_stokes_fvca8")
    parser.add_argument("--case", required=True, choices=("bercovier_engelman_2d", "taylor_green_3d"))
    parser.add_argument("--resolution", type=int, action="append", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--plot", default=None)
    parser.add_argument("--omp-threads", default="1")
    parser.add_argument("--krylov", action="store_true", help="Use BiCGStab instead of the small dense fallback")
    parser.add_argument("--skip-solve", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = Path(args.exe)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "python")
    env["OMP_NUM_THREADS"] = args.omp_threads
    if args.krylov:
        env["SFEM_DENSE_SOLVE"] = "0"

    levels = []
    for resolution in args.resolution:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        run_dir = out_root / ("n%d" % resolution)
        h = 1.0 / resolution
        levels.extend(["--level", "n%d:%.17g:%s:%s" % (resolution, h, run_dir, run_dir)])
        if not args.skip_solve:
            cmd = [str(exe), args.case, str(resolution), str(run_dir)]
            subprocess.run(cmd, check=True, env=env)

    csv_path = Path(args.csv) if args.csv else out_root / ("%s_convergence.csv" % args.case)
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_stokes_convergence.py")),
        "--case",
        args.case,
        *levels,
        "--out",
        str(csv_path),
    ]
    subprocess.run(cmd, check=True, env=env)

    if args.plot:
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("plot_convergence.py")),
            str(csv_path),
            "--out",
            args.plot,
        ]
        subprocess.run(cmd, check=True, env=env)

    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
